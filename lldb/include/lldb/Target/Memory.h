//===-- Memory.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Memory_h_
#define liblldb_Memory_h_

// C Includes
// C++ Includes
#include <map>
#include <vector>

// Other libraries and framework includes

// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/RangeMap.h"
#include "lldb/Host/Mutex.h"

namespace lldb_private {
    //----------------------------------------------------------------------
    // A class to track memory that was read from a live process between 
    // runs. 
    //----------------------------------------------------------------------
    class MemoryCache
    {
    public:
        //------------------------------------------------------------------
        // Constructors and Destructors
        //------------------------------------------------------------------
        MemoryCache (Process &process);
        
        ~MemoryCache ();
        
        void
        Clear();
        
        void
        Flush (lldb::addr_t addr, size_t size);
        
        size_t
        Read (lldb::addr_t addr, 
              void *dst, 
              size_t dst_len,
              Error &error);
        
        uint32_t
        GetMemoryCacheLineSize() const
        {
            return m_cache_line_byte_size ;
        }
        
        void
        AddInvalidRange (lldb::addr_t base_addr, lldb::addr_t byte_size);

        bool
        RemoveInvalidRange (lldb::addr_t base_addr, lldb::addr_t byte_size);

    protected:
        typedef std::map<lldb::addr_t, lldb::DataBufferSP> BlockMap;
        typedef RangeArray<lldb::addr_t, lldb::addr_t, 4> InvalidRanges;
        //------------------------------------------------------------------
        // Classes that inherit from MemoryCache can see and modify these
        //------------------------------------------------------------------
        Process &m_process;
        uint32_t m_cache_line_byte_size;
        Mutex m_mutex;
        BlockMap m_cache;
        InvalidRanges m_invalid_ranges;
    private:
        DISALLOW_COPY_AND_ASSIGN (MemoryCache);
    };

    
    class AllocatedBlock
    {
    public:
        AllocatedBlock (lldb::addr_t addr, 
                        uint32_t byte_size, 
                        uint32_t permissions,
                        uint32_t chunk_size);
        
        ~AllocatedBlock ();
        
        lldb::addr_t
        ReserveBlock (uint32_t size);
        
        bool
        FreeBlock (lldb::addr_t addr);
        
        lldb::addr_t
        GetBaseAddress () const
        {
            return m_addr;
        }
        
        uint32_t
        GetByteSize () const
        {
            return m_byte_size;
        }

        uint32_t
        GetPermissions () const
        {
            return m_permissions;
        }

        uint32_t
        GetChunkSize () const
        {
            return m_chunk_size;
        }

        bool
        Contains (lldb::addr_t addr) const
        {
            return ((addr >= m_addr) && addr < (m_addr + m_byte_size));
        }
    protected:
        uint32_t
        TotalChunks () const
        {
            return m_byte_size / m_chunk_size;
        }
        
        uint32_t 
        CalculateChunksNeededForSize (uint32_t size) const
        {
            return (size + m_chunk_size - 1) / m_chunk_size;
        }
        const lldb::addr_t m_addr;    // Base address of this block of memory
        const uint32_t m_byte_size;   // 4GB of chunk should be enough...
        const uint32_t m_permissions; // Permissions for this memory (logical OR of lldb::Permissions bits)
        const uint32_t m_chunk_size;  // The size of chunks that the memory at m_addr is divied up into
        typedef std::map<uint32_t, uint32_t> OffsetToChunkSize;
        OffsetToChunkSize m_offset_to_chunk_size;
    };
    

    //----------------------------------------------------------------------
    // A class that can track allocated memory and give out allocated memory
    // without us having to make an allocate/deallocate call every time we
    // need some memory in a process that is being debugged.
    //----------------------------------------------------------------------
    class AllocatedMemoryCache
    {
    public:
        //------------------------------------------------------------------
        // Constructors and Destructors
        //------------------------------------------------------------------
        AllocatedMemoryCache (Process &process);
        
        ~AllocatedMemoryCache ();
        
        void
        Clear();

        lldb::addr_t
        AllocateMemory (size_t byte_size, 
                        uint32_t permissions, 
                        Error &error);

        bool
        DeallocateMemory (lldb::addr_t ptr);
        
    protected:
        typedef std::shared_ptr<AllocatedBlock> AllocatedBlockSP;

        AllocatedBlockSP
        AllocatePage (uint32_t byte_size, 
                      uint32_t permissions, 
                      uint32_t chunk_size, 
                      Error &error);


        //------------------------------------------------------------------
        // Classes that inherit from MemoryCache can see and modify these
        //------------------------------------------------------------------
        Process &m_process;
        Mutex m_mutex;
        typedef std::multimap<uint32_t, AllocatedBlockSP> PermissionsToBlockMap;
        PermissionsToBlockMap m_memory_map;
        
    private:
        DISALLOW_COPY_AND_ASSIGN (AllocatedMemoryCache);
    };

} // namespace lldb_private

#endif  // liblldb_Memory_h_
