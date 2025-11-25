//! Memory Management Module
//!
//! Safe Rust abstractions for kernel memory operations:
//! - Safe ioremap/iounmap wrappers with automatic cleanup
//! - Chunk management for 360MB memory region
//! - MMIO access patterns with bounds checking
//! - Resource lifecycle management

use crate::{DsmilError, DsmilResult};
use crate::constants::*;
use core::ptr::NonNull;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};

/// Memory chunk for managing large MMIO regions
#[derive(Debug)]
pub struct MemoryChunk {
    base_addr: u64,
    size: usize,
    mapped: Option<NonNull<u8>>,
    _phantom: PhantomData<*mut u8>, // !Send + !Sync for kernel memory
}

impl MemoryChunk {
    /// Create new memory chunk (unmapped initially)
    pub fn new(base_addr: u64, size: usize) -> Self {
        MemoryChunk {
            base_addr,
            size,
            mapped: None,
            _phantom: PhantomData,
        }
    }
    
    /// Map the memory chunk
    pub fn map(&mut self) -> DsmilResult<()> {
        if self.mapped.is_some() {
            return Err(DsmilError::Busy);
        }
        
        // Check alignment and size constraints
        if self.base_addr & 0xFFF != 0 {
            return Err(DsmilError::InvalidDevice);
        }
        
        if self.size == 0 || self.size > DSMIL_MEMORY_SIZE {
            return Err(DsmilError::InvalidDevice);
        }
        
        // Call kernel ioremap through FFI
        let mapped_ptr = unsafe { kernel_ioremap(self.base_addr, self.size) };
        
        if mapped_ptr.is_null() {
            return Err(DsmilError::OutOfMemory);
        }
        
        self.mapped = NonNull::new(mapped_ptr);
        Ok(())
    }
    
    /// Unmap the memory chunk
    pub fn unmap(&mut self) {
        if let Some(ptr) = self.mapped.take() {
            unsafe {
                kernel_iounmap(ptr.as_ptr(), self.size);
            }
        }
    }
    
    /// Get mapped pointer (if mapped)
    pub fn as_ptr(&self) -> Option<NonNull<u8>> {
        self.mapped
    }
    
    /// Check if chunk is mapped
    pub fn is_mapped(&self) -> bool {
        self.mapped.is_some()
    }
    
    /// Get base address
    pub fn base_addr(&self) -> u64 {
        self.base_addr
    }
    
    /// Get size
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Safe read from chunk with bounds checking
    pub fn read_u32(&self, offset: usize) -> DsmilResult<u32> {
        if offset + 4 > self.size {
            return Err(DsmilError::InvalidDevice);
        }
        
        let ptr = self.mapped.ok_or(DsmilError::NotFound)?;
        
        unsafe {
            let addr = ptr.as_ptr().add(offset) as *const u32;
            Ok(addr.read_volatile())
        }
    }
    
    /// Safe write to chunk with bounds checking
    pub fn write_u32(&self, offset: usize, value: u32) -> DsmilResult<()> {
        if offset + 4 > self.size {
            return Err(DsmilError::InvalidDevice);
        }
        
        let ptr = self.mapped.ok_or(DsmilError::NotFound)?;
        
        unsafe {
            let addr = ptr.as_ptr().add(offset) as *mut u32;
            addr.write_volatile(value);
        }
        
        Ok(())
    }
    
    /// Read block of data safely
    pub fn read_block(&self, offset: usize, buffer: &mut [u8]) -> DsmilResult<()> {
        if offset + buffer.len() > self.size {
            return Err(DsmilError::InvalidDevice);
        }
        
        let ptr = self.mapped.ok_or(DsmilError::NotFound)?;
        
        unsafe {
            let src = ptr.as_ptr().add(offset);
            core::ptr::copy_nonoverlapping(src, buffer.as_mut_ptr(), buffer.len());
        }
        
        Ok(())
    }
    
    /// Write block of data safely
    pub fn write_block(&self, offset: usize, data: &[u8]) -> DsmilResult<()> {
        if offset + data.len() > self.size {
            return Err(DsmilError::InvalidDevice);
        }
        
        let ptr = self.mapped.ok_or(DsmilError::NotFound)?;
        
        unsafe {
            let dst = ptr.as_ptr().add(offset);
            core::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        
        Ok(())
    }
}

impl Drop for MemoryChunk {
    fn drop(&mut self) {
        self.unmap();
    }
}

/// MMIO Region with safe access patterns (max 90 chunks for 360MB)
pub struct MmioRegion {
    chunks: [Option<MemoryChunk>; 90], // 360MB / 4MB chunks = 90 max
    chunk_count: usize,
    total_size: usize,
    base_addr: u64,
}

impl MmioRegion {
    /// Create new MMIO region with chunked mapping
    pub fn new(base_addr: u64, size: usize) -> DsmilResult<Self> {
        if size == 0 {
            return Err(DsmilError::InvalidDevice);
        }
        
        // Calculate number of chunks needed
        let chunk_count = (size + DSMIL_CHUNK_SIZE - 1) / DSMIL_CHUNK_SIZE;
        if chunk_count > 90 {
            return Err(DsmilError::OutOfMemory); // Too many chunks needed
        }
        
        let mut chunks: [Option<MemoryChunk>; 90] = [const { None }; 90];
        
        // Create chunks
        for i in 0..chunk_count {
            let chunk_base = base_addr + (i * DSMIL_CHUNK_SIZE) as u64;
            let chunk_size = if i == chunk_count - 1 {
                // Last chunk might be smaller
                size - (i * DSMIL_CHUNK_SIZE)
            } else {
                DSMIL_CHUNK_SIZE
            };
            
            chunks[i] = Some(MemoryChunk::new(chunk_base, chunk_size));
        }
        
        Ok(MmioRegion {
            chunks,
            chunk_count,
            total_size: size,
            base_addr,
        })
    }
    
    /// Map all chunks
    pub fn map_all(&mut self) -> DsmilResult<()> {
        for i in 0..self.chunk_count {
            if let Some(ref mut chunk) = self.chunks[i] {
                chunk.map()?;
            }
        }
        Ok(())
    }
    
    /// Map specific chunk by index
    pub fn map_chunk(&mut self, chunk_index: usize) -> DsmilResult<()> {
        if chunk_index >= self.chunk_count {
            return Err(DsmilError::InvalidDevice);
        }
        
        if let Some(ref mut chunk) = self.chunks[chunk_index] {
            chunk.map()
        } else {
            Err(DsmilError::NotFound)
        }
    }
    
    /// Get chunk containing the given offset
    fn get_chunk_for_offset(&self, offset: usize) -> DsmilResult<(usize, usize)> {
        if offset >= self.total_size {
            return Err(DsmilError::InvalidDevice);
        }
        
        let chunk_index = offset / DSMIL_CHUNK_SIZE;
        let chunk_offset = offset % DSMIL_CHUNK_SIZE;
        
        Ok((chunk_index, chunk_offset))
    }
    
    /// Read u32 from MMIO region with on-demand mapping
    pub fn read_u32(&mut self, offset: usize) -> DsmilResult<u32> {
        let (chunk_index, chunk_offset) = self.get_chunk_for_offset(offset)?;
        
        if let Some(ref mut chunk) = self.chunks[chunk_index] {
            // Ensure chunk is mapped
            if !chunk.is_mapped() {
                chunk.map()?;
            }
            
            chunk.read_u32(chunk_offset)
        } else {
            Err(DsmilError::NotFound)
        }
    }
    
    /// Write u32 to MMIO region with on-demand mapping
    pub fn write_u32(&mut self, offset: usize, value: u32) -> DsmilResult<()> {
        let (chunk_index, chunk_offset) = self.get_chunk_for_offset(offset)?;
        
        if let Some(ref mut chunk) = self.chunks[chunk_index] {
            // Ensure chunk is mapped
            if !chunk.is_mapped() {
                chunk.map()?;
            }
            
            chunk.write_u32(chunk_offset, value)
        } else {
            Err(DsmilError::NotFound)
        }
    }
    
    /// Read block from MMIO region (may span chunks)
    pub fn read_block(&mut self, offset: usize, buffer: &mut [u8]) -> DsmilResult<()> {
        if offset + buffer.len() > self.total_size {
            return Err(DsmilError::InvalidDevice);
        }
        
        let mut remaining = buffer.len();
        let mut current_offset = offset;
        let mut buffer_pos = 0;
        
        while remaining > 0 {
            let (chunk_index, chunk_offset) = self.get_chunk_for_offset(current_offset)?;
            
            if let Some(ref mut chunk) = self.chunks[chunk_index] {
                // Ensure chunk is mapped
                if !chunk.is_mapped() {
                    chunk.map()?;
                }
                
                // Calculate how much we can read from this chunk
                let chunk_remaining = chunk.size() - chunk_offset;
                let to_read = remaining.min(chunk_remaining);
                
                // Read from chunk
                let chunk_buffer = &mut buffer[buffer_pos..buffer_pos + to_read];
                chunk.read_block(chunk_offset, chunk_buffer)?;
                
                remaining -= to_read;
                current_offset += to_read;
                buffer_pos += to_read;
            } else {
                return Err(DsmilError::NotFound);
            }
        }
        
        Ok(())
    }
    
    /// Write block to MMIO region (may span chunks)
    pub fn write_block(&mut self, offset: usize, data: &[u8]) -> DsmilResult<()> {
        if offset + data.len() > self.total_size {
            return Err(DsmilError::InvalidDevice);
        }
        
        let mut remaining = data.len();
        let mut current_offset = offset;
        let mut data_pos = 0;
        
        while remaining > 0 {
            let (chunk_index, chunk_offset) = self.get_chunk_for_offset(current_offset)?;
            
            if let Some(ref mut chunk) = self.chunks[chunk_index] {
                // Ensure chunk is mapped
                if !chunk.is_mapped() {
                    chunk.map()?;
                }
                
                // Calculate how much we can write to this chunk
                let chunk_remaining = chunk.size() - chunk_offset;
                let to_write = remaining.min(chunk_remaining);
                
                // Write to chunk
                let chunk_data = &data[data_pos..data_pos + to_write];
                chunk.write_block(chunk_offset, chunk_data)?;
                
                remaining -= to_write;
                current_offset += to_write;
                data_pos += to_write;
            } else {
                return Err(DsmilError::NotFound);
            }
        }
        
        Ok(())
    }
    
    /// Get total size
    pub fn size(&self) -> usize {
        self.total_size
    }
    
    /// Get base address
    pub fn base_addr(&self) -> u64 {
        self.base_addr
    }
    
    /// Get number of chunks
    pub fn chunk_count(&self) -> usize {
        self.chunk_count
    }
    
    /// Check if all chunks are mapped
    pub fn all_chunks_mapped(&self) -> bool {
        for i in 0..self.chunk_count {
            if let Some(ref chunk) = self.chunks[i] {
                if !chunk.is_mapped() {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }
    
    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let mut mapped_chunks = 0;
        for i in 0..self.chunk_count {
            if let Some(ref chunk) = self.chunks[i] {
                if chunk.is_mapped() {
                    mapped_chunks += 1;
                }
            }
        }
        let mapped_size = mapped_chunks * DSMIL_CHUNK_SIZE;
        
        MemoryStats {
            total_size: self.total_size,
            mapped_size,
            chunk_count: self.chunk_count,
            mapped_chunks,
        }
    }
    
    /// Unmap unused chunks to free memory
    pub fn cleanup_unused(&mut self) {
        // In a real implementation, we'd track access patterns
        // and unmap chunks that haven't been used recently
        // For now, this is a placeholder
    }
}

impl Drop for MmioRegion {
    fn drop(&mut self) {
        // Chunks automatically unmap themselves in their Drop impl
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    pub total_size: usize,
    pub mapped_size: usize,
    pub chunk_count: usize,
    pub mapped_chunks: usize,
}

/// Memory pool for managing multiple MMIO regions (max 10 regions)
pub struct MemoryPool {
    regions: [Option<MmioRegion>; 10],
    region_count: usize,
    total_mapped: usize,
    max_mapped: usize,
}

impl MemoryPool {
    /// Create new memory pool
    pub fn new(max_mapped_mb: usize) -> Self {
        MemoryPool {
            regions: [const { None }; 10],
            region_count: 0,
            total_mapped: 0,
            max_mapped: max_mapped_mb * 1024 * 1024,
        }
    }
    
    /// Add region to pool
    pub fn add_region(&mut self, base_addr: u64, size: usize) -> DsmilResult<usize> {
        if self.region_count >= 10 {
            return Err(DsmilError::OutOfMemory);
        }
        
        let region = MmioRegion::new(base_addr, size)?;
        let region_id = self.region_count;
        self.regions[region_id] = Some(region);
        self.region_count += 1;
        Ok(region_id)
    }
    
    /// Get mutable region reference
    pub fn region_mut(&mut self, region_id: usize) -> Option<&mut MmioRegion> {
        if region_id >= self.region_count {
            None
        } else {
            self.regions[region_id].as_mut()
        }
    }
    
    /// Get region reference
    pub fn region(&self, region_id: usize) -> Option<&MmioRegion> {
        if region_id >= self.region_count {
            None
        } else {
            self.regions[region_id].as_ref()
        }
    }
    
    /// Check if we can map more memory
    fn can_map(&self, additional_size: usize) -> bool {
        self.total_mapped + additional_size <= self.max_mapped
    }
    
    /// Update mapped memory tracking
    fn update_mapped_size(&mut self) {
        self.total_mapped = 0;
        for i in 0..self.region_count {
            if let Some(ref region) = self.regions[i] {
                self.total_mapped += region.memory_stats().mapped_size;
            }
        }
    }
    
    /// Get pool statistics
    pub fn pool_stats(&self) -> PoolStats {
        let mut total_size = 0;
        let mut mapped_size = 0;
        
        for i in 0..self.region_count {
            if let Some(ref region) = self.regions[i] {
                total_size += region.size();
                mapped_size += region.memory_stats().mapped_size;
            }
        }
        
        PoolStats {
            region_count: self.region_count,
            total_size,
            mapped_size,
            max_mapped: self.max_mapped,
        }
    }
}

/// Pool statistics
#[derive(Debug, Clone, Copy)]
pub struct PoolStats {
    pub region_count: usize,
    pub total_size: usize,
    pub mapped_size: usize,
    pub max_mapped: usize,
}

/// External C functions for kernel memory operations
extern "C" {
    /// Map I/O memory region
    fn kernel_ioremap(phys_addr: u64, size: usize) -> *mut u8;
    
    /// Unmap I/O memory region
    fn kernel_iounmap(addr: *mut u8, size: usize);
    
    /// Check if memory region is valid
    fn kernel_mem_valid(phys_addr: u64, size: usize) -> bool;
    
    /// Get page size
    fn kernel_page_size() -> usize;
}

/// Helper functions for memory management

/// Try multiple base addresses for DSMIL region discovery
pub fn discover_dsmil_region(size: usize) -> DsmilResult<u64> {
    const BASE_ADDRESSES: [u64; 5] = [
        DSMIL_PRIMARY_BASE,
        DSMIL_JRTC1_BASE,
        DSMIL_EXTENDED_BASE,
        DSMIL_PLATFORM_BASE,
        DSMIL_HIGH_BASE,
    ];
    
    for &base_addr in &BASE_ADDRESSES {
        // Check if this region is valid
        if unsafe { kernel_mem_valid(base_addr, size) } {
            // Try a test mapping
            let test_ptr = unsafe { kernel_ioremap(base_addr, 4096) }; // Test with single page
            if !test_ptr.is_null() {
                unsafe { kernel_iounmap(test_ptr, 4096) };
                return Ok(base_addr);
            }
        }
    }
    
    Err(DsmilError::NotFound)
}

/// Validate memory alignment
pub fn validate_alignment(addr: u64, size: usize) -> DsmilResult<()> {
    let page_size = unsafe { kernel_page_size() };
    
    if (addr as usize) % page_size != 0 {
        return Err(DsmilError::InvalidDevice);
    }
    
    if size % page_size != 0 {
        return Err(DsmilError::InvalidDevice);
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_chunk_creation() {
        let chunk = MemoryChunk::new(0x52000000, 4096);
        assert_eq!(chunk.base_addr(), 0x52000000);
        assert_eq!(chunk.size(), 4096);
        assert!(!chunk.is_mapped());
    }
    
    #[test]
    fn test_mmio_region_chunking() {
        let region = MmioRegion::new(0x52000000, DSMIL_CHUNK_SIZE * 2).unwrap();
        assert_eq!(region.chunk_count(), 2);
        assert_eq!(region.size(), DSMIL_CHUNK_SIZE * 2);
    }
    
    #[test]
    fn test_chunk_offset_calculation() {
        let region = MmioRegion::new(0x52000000, DSMIL_CHUNK_SIZE * 2).unwrap();
        
        let (chunk_idx, chunk_offset) = region.get_chunk_for_offset(0).unwrap();
        assert_eq!(chunk_idx, 0);
        assert_eq!(chunk_offset, 0);
        
        let (chunk_idx, chunk_offset) = region.get_chunk_for_offset(DSMIL_CHUNK_SIZE + 100).unwrap();
        assert_eq!(chunk_idx, 1);
        assert_eq!(chunk_offset, 100);
    }
    
    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(100); // 100MB max
        let region_id = pool.add_region(0x52000000, DSMIL_CHUNK_SIZE).unwrap();
        assert_eq!(region_id, 0);
        
        let stats = pool.pool_stats();
        assert_eq!(stats.region_count, 1);
        assert_eq!(stats.total_size, DSMIL_CHUNK_SIZE);
    }
}