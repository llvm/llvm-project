//===-- DataExtractor.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <stddef.h>

#include <bitset>
#include <limits>
#include <sstream>
#include <string>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/MathExtras.h"

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/UUID.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Host/Endian.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

static inline uint16_t 
ReadInt16(const unsigned char* ptr, unsigned offset) 
{
    return *(uint16_t *)(ptr + offset);
}
static inline uint32_t
ReadInt32 (const unsigned char* ptr, unsigned offset) 
{
    return *(uint32_t *)(ptr + offset);
}

static inline uint64_t 
ReadInt64(const unsigned char* ptr, unsigned offset) 
{
    return *(uint64_t *)(ptr + offset);
}

static inline uint16_t
ReadSwapInt16(const unsigned char* ptr, unsigned offset) 
{
    return llvm::ByteSwap_16(*(uint16_t *)(ptr + offset));
}

static inline uint32_t
ReadSwapInt32 (const unsigned char* ptr, unsigned offset) 
{
    return llvm::ByteSwap_32(*(uint32_t *)(ptr + offset));
}
static inline uint64_t 
ReadSwapInt64(const unsigned char* ptr, unsigned offset) 
{
  return llvm::ByteSwap_64(*(uint64_t *)(ptr + offset));
}

#define NON_PRINTABLE_CHAR '.'
//----------------------------------------------------------------------
// Default constructor.
//----------------------------------------------------------------------
DataExtractor::DataExtractor () :
    m_start     (NULL),
    m_end       (NULL),
    m_byte_order(lldb::endian::InlHostByteOrder()),
    m_addr_size (4),
    m_data_sp   ()
{
}

//----------------------------------------------------------------------
// This constructor allows us to use data that is owned by someone else.
// The data must stay around as long as this object is valid.
//----------------------------------------------------------------------
DataExtractor::DataExtractor (const void* data, uint32_t length, ByteOrder endian, uint8_t addr_size) :
    m_start     ((uint8_t*)data),
    m_end       ((uint8_t*)data + length),
    m_byte_order(endian),
    m_addr_size (addr_size),
    m_data_sp   ()
{
}

//----------------------------------------------------------------------
// Make a shared pointer reference to the shared data in "data_sp" and
// set the endian swapping setting to "swap", and the address size to
// "addr_size". The shared data reference will ensure the data lives
// as long as any DataExtractor objects exist that have a reference to
// this data.
//----------------------------------------------------------------------
DataExtractor::DataExtractor (const DataBufferSP& data_sp, ByteOrder endian, uint8_t addr_size) :
    m_start     (NULL),
    m_end       (NULL),
    m_byte_order(endian),
    m_addr_size (addr_size),
    m_data_sp   ()
{
    SetData (data_sp);
}

//----------------------------------------------------------------------
// Initialize this object with a subset of the data bytes in "data".
// If "data" contains shared data, then a reference to this shared
// data will added and the shared data will stay around as long
// as any object contains a reference to that data. The endian
// swap and address size settings are copied from "data".
//----------------------------------------------------------------------
DataExtractor::DataExtractor (const DataExtractor& data, uint32_t offset, uint32_t length) :
    m_start(NULL),
    m_end(NULL),
    m_byte_order(data.m_byte_order),
    m_addr_size(data.m_addr_size),
    m_data_sp()
{
    if (data.ValidOffset(offset))
    {
        uint32_t bytes_available = data.GetByteSize() - offset;
        if (length > bytes_available)
            length = bytes_available;
        SetData(data, offset, length);
    }
}

DataExtractor::DataExtractor (const DataExtractor& rhs) :
    m_start (rhs.m_start),
    m_end (rhs.m_end),
    m_byte_order (rhs.m_byte_order),
    m_addr_size (rhs.m_addr_size),
    m_data_sp (rhs.m_data_sp)
{
}

//----------------------------------------------------------------------
// Assignment operator
//----------------------------------------------------------------------
const DataExtractor&
DataExtractor::operator= (const DataExtractor& rhs)
{
    if (this != &rhs)
    {
        m_start = rhs.m_start;
        m_end = rhs.m_end;
        m_byte_order = rhs.m_byte_order;
        m_addr_size = rhs.m_addr_size;
        m_data_sp = rhs.m_data_sp;
    }
    return *this;
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
DataExtractor::~DataExtractor ()
{
}

//------------------------------------------------------------------
// Clears the object contents back to a default invalid state, and
// release any references to shared data that this object may
// contain.
//------------------------------------------------------------------
void
DataExtractor::Clear ()
{
    m_start = NULL;
    m_end = NULL;
    m_byte_order = lldb::endian::InlHostByteOrder();
    m_addr_size = 4;
    m_data_sp.reset();
}

//------------------------------------------------------------------
// If this object contains shared data, this function returns the
// offset into that shared data. Else zero is returned.
//------------------------------------------------------------------
size_t
DataExtractor::GetSharedDataOffset () const
{
    if (m_start != NULL)
    {
        const DataBuffer * data = m_data_sp.get();
        if (data != NULL)
        {
            const uint8_t * data_bytes = data->GetBytes();
            if (data_bytes != NULL)
            {
                assert(m_start >= data_bytes);
                return m_start - data_bytes;
            }
        }
    }
    return 0;
}

//------------------------------------------------------------------
// Returns true if there are LENGTH bytes availabe starting OFFSET
// into the data that is in this object.
//------------------------------------------------------------------
bool
DataExtractor::ValidOffsetForDataOfSize (uint32_t offset, uint32_t length) const
{
    size_t size = GetByteSize();
    if (offset >= size)
        return false;   // offset isn't valid

    if (length == 0)
        return true;    // No bytes requested at this offset, return true

    // If we flip the bits in offset we can figure out how
    // many bytes we have left before "offset + length"
    // could overflow when doing unsigned arithmetic.
    if (length > ~offset)
        return false;   // unsigned overflow

    // Make sure "offset + length" is a valid offset as well.
    // length must be greater than zero for this to be a
    // valid expression, and we have already checked for this.
    return ((offset + length) <= size);
}

//----------------------------------------------------------------------
// Set the data with which this object will extract from to data
// starting at BYTES and set the length of the data to LENGTH bytes
// long. The data is externally owned must be around at least as
// long as this object points to the data. No copy of the data is
// made, this object just refers to this data and can extract from
// it. If this object refers to any shared data upon entry, the
// reference to that data will be released. Is SWAP is set to true,
// any data extracted will be endian swapped.
//----------------------------------------------------------------------
uint32_t
DataExtractor::SetData (const void *bytes, uint32_t length, ByteOrder endian)
{
    m_byte_order = endian;
    m_data_sp.reset();
    if (bytes == NULL || length == 0)
    {
        m_start = NULL;
        m_end = NULL;
    }
    else
    {
        m_start = (uint8_t *)bytes;
        m_end = m_start + length;
    }
    return GetByteSize();
}

//----------------------------------------------------------------------
// Assign the data for this object to be a subrange in "data"
// starting "data_offset" bytes into "data" and ending "data_length"
// bytes later. If "data_offset" is not a valid offset into "data",
// then this object will contain no bytes. If "data_offset" is
// within "data" yet "data_length" is too large, the length will be
// capped at the number of bytes remaining in "data". If "data"
// contains a shared pointer to other data, then a ref counted
// pointer to that data will be made in this object. If "data"
// doesn't contain a shared pointer to data, then the bytes referred
// to in "data" will need to exist at least as long as this object
// refers to those bytes. The address size and endian swap settings
// are copied from the current values in "data".
//----------------------------------------------------------------------
uint32_t
DataExtractor::SetData (const DataExtractor& data, uint32_t data_offset, uint32_t data_length)
{
    m_addr_size = data.m_addr_size;
    // If "data" contains shared pointer to data, then we can use that
    if (data.m_data_sp.get())
    {
        m_byte_order = data.m_byte_order;
        return SetData(data.m_data_sp, data.GetSharedDataOffset() + data_offset, data_length);
    }

    // We have a DataExtractor object that just has a pointer to bytes
    if (data.ValidOffset(data_offset))
    {
        if (data_length > data.GetByteSize() - data_offset)
            data_length = data.GetByteSize() - data_offset;
        return SetData (data.GetDataStart() + data_offset, data_length, data.GetByteOrder());
    }
    return 0;
}

//----------------------------------------------------------------------
// Assign the data for this object to be a subrange of the shared
// data in "data_sp" starting "data_offset" bytes into "data_sp"
// and ending "data_length" bytes later. If "data_offset" is not
// a valid offset into "data_sp", then this object will contain no
// bytes. If "data_offset" is within "data_sp" yet "data_length" is
// too large, the length will be capped at the number of bytes
// remaining in "data_sp". A ref counted pointer to the data in
// "data_sp" will be made in this object IF the number of bytes this
// object refers to in greater than zero (if at least one byte was
// available starting at "data_offset") to ensure the data stays
// around as long as it is needed. The address size and endian swap
// settings will remain unchanged from their current settings.
//----------------------------------------------------------------------
uint32_t
DataExtractor::SetData (const DataBufferSP& data_sp, uint32_t data_offset, uint32_t data_length)
{
    m_start = m_end = NULL;

    if (data_length > 0)
    {
        m_data_sp = data_sp;
        if (data_sp.get())
        {
            const size_t data_size = data_sp->GetByteSize();
            if (data_offset < data_size)
            {
                m_start = data_sp->GetBytes() + data_offset;
                const size_t bytes_left = data_size - data_offset;
                // Cap the length of we asked for too many
                if (data_length <= bytes_left)
                    m_end = m_start + data_length;  // We got all the bytes we wanted
                else
                    m_end = m_start + bytes_left;   // Not all the bytes requested were available in the shared data
            }
        }
    }

    uint32_t new_size = GetByteSize();

    // Don't hold a shared pointer to the data buffer if we don't share
    // any valid bytes in the shared buffer.
    if (new_size == 0)
        m_data_sp.reset();

    return new_size;
}

//----------------------------------------------------------------------
// Extract a single unsigned char from the binary data and update
// the offset pointed to by "offset_ptr".
//
// RETURNS the byte that was extracted, or zero on failure.
//----------------------------------------------------------------------
uint8_t
DataExtractor::GetU8 (uint32_t *offset_ptr) const
{
    uint8_t val = 0;
    if ( m_start < m_end )
    {
        val = m_start[*offset_ptr];
        *offset_ptr += sizeof(val);
    }
    return val;
}

//----------------------------------------------------------------------
// Extract "count" unsigned chars from the binary data and update the
// offset pointed to by "offset_ptr". The extracted data is copied into
// "dst".
//
// RETURNS the non-NULL buffer pointer upon successful extraction of
// all the requested bytes, or NULL when the data is not available in
// the buffer due to being out of bounds, or unsufficient data.
//----------------------------------------------------------------------
void *
DataExtractor::GetU8 (uint32_t *offset_ptr, void *dst, uint32_t count) const
{
    register uint32_t offset = *offset_ptr;

    if ((count > 0) && ValidOffsetForDataOfSize(offset, count) )
    {
        // Copy the data into the buffer
        memcpy (dst, m_start + offset, count);
        // Advance the offset
        *offset_ptr += count;
        // Return a non-NULL pointer to the converted data as an indicator of success
        return dst;
    }
    return NULL;
}

//----------------------------------------------------------------------
// Extract a single uint16_t from the data and update the offset
// pointed to by "offset_ptr".
//
// RETURNS the uint16_t that was extracted, or zero on failure.
//----------------------------------------------------------------------
uint16_t
DataExtractor::GetU16 (uint32_t *offset_ptr) const
{
    uint16_t val = 0;
    register uint32_t offset = *offset_ptr;
    if ( ValidOffsetForDataOfSize(offset, sizeof(val)) )
    {
        if (m_byte_order != lldb::endian::InlHostByteOrder())
            val = ReadSwapInt16(m_start, offset);
        else
            val = ReadInt16 (m_start, offset);

        // Advance the offset
        *offset_ptr += sizeof(val);
    }
    return val;
}

uint16_t
DataExtractor::GetU16_unchecked (uint32_t *offset_ptr) const
{
    uint16_t val = (m_byte_order == lldb::endian::InlHostByteOrder()) ? 
                        ReadInt16 (m_start, *offset_ptr) :
                        ReadSwapInt16(m_start, *offset_ptr);
    *offset_ptr += sizeof(val);
    return val;
}

uint32_t
DataExtractor::GetU32_unchecked (uint32_t *offset_ptr) const
{
    uint32_t val = (m_byte_order == lldb::endian::InlHostByteOrder()) ? 
                        ReadInt32 (m_start, *offset_ptr) :
                        ReadSwapInt32 (m_start, *offset_ptr);
    *offset_ptr += sizeof(val);
    return val;
}

uint64_t
DataExtractor::GetU64_unchecked (uint32_t *offset_ptr) const
{
    uint64_t val = (m_byte_order == lldb::endian::InlHostByteOrder()) ? 
                        ReadInt64 (m_start, *offset_ptr) :
                        ReadSwapInt64 (m_start, *offset_ptr);
    *offset_ptr += sizeof(val);
    return val;
}


//----------------------------------------------------------------------
// Extract "count" uint16_t values from the binary data and update
// the offset pointed to by "offset_ptr". The extracted data is
// copied into "dst".
//
// RETURNS the non-NULL buffer pointer upon successful extraction of
// all the requested bytes, or NULL when the data is not available
// in the buffer due to being out of bounds, or unsufficient data.
//----------------------------------------------------------------------
void *
DataExtractor::GetU16 (uint32_t *offset_ptr, void *void_dst, uint32_t count) const
{
    uint16_t *dst = (uint16_t *)void_dst;
    const size_t value_size = sizeof(*dst);
    register uint32_t offset = *offset_ptr;

    if ((count > 0) && ValidOffsetForDataOfSize(offset, value_size * count) )
    {
        uint16_t *value_ptr;
        uint16_t *end = dst + count;
        if (m_byte_order != lldb::endian::InlHostByteOrder())
        {
            for (value_ptr = dst; value_ptr < end; ++value_ptr, offset += value_size)
                *value_ptr = ReadSwapInt16 (m_start, offset);
        }
        else
        {
            for (value_ptr = dst; value_ptr < end; ++value_ptr, offset += value_size)
                *value_ptr = ReadInt16 (m_start, offset);
        }

        // Advance the offset
        *offset_ptr = offset;
        // Return a non-NULL pointer to the converted data as an indicator of success
        return dst;
    }
    return NULL;
}

//----------------------------------------------------------------------
// Extract a single uint32_t from the data and update the offset
// pointed to by "offset_ptr".
//
// RETURNS the uint32_t that was extracted, or zero on failure.
//----------------------------------------------------------------------
uint32_t
DataExtractor::GetU32 (uint32_t *offset_ptr) const
{
    uint32_t val = 0;
    register uint32_t offset = *offset_ptr;

    if ( ValidOffsetForDataOfSize(offset, sizeof(val)) )
    {
        if (m_byte_order != lldb::endian::InlHostByteOrder())
            val = ReadSwapInt32 (m_start, offset);
        else
            val = ReadInt32 (m_start, offset);

        // Advance the offset
        *offset_ptr += sizeof(val);
    }
    return val;
}

//----------------------------------------------------------------------
// Extract "count" uint32_t values from the binary data and update
// the offset pointed to by "offset_ptr". The extracted data is
// copied into "dst".
//
// RETURNS the non-NULL buffer pointer upon successful extraction of
// all the requested bytes, or NULL when the data is not available
// in the buffer due to being out of bounds, or unsufficient data.
//----------------------------------------------------------------------
void *
DataExtractor::GetU32 (uint32_t *offset_ptr, void *void_dst, uint32_t count) const
{
    uint32_t *dst = (uint32_t *)void_dst;
    const size_t value_size = sizeof(*dst);
    register uint32_t offset = *offset_ptr;

    if ((count > 0) && ValidOffsetForDataOfSize(offset, value_size * count))
    {
        uint32_t *value_ptr;
        uint32_t *end = dst + count;
        if (m_byte_order != lldb::endian::InlHostByteOrder())
        {
            for (value_ptr = dst; value_ptr < end; ++value_ptr, offset += value_size)
                *value_ptr = ReadSwapInt32 (m_start, offset);

        }
        else
        {
            for (value_ptr = dst; value_ptr < end; ++value_ptr, offset += value_size)
                *value_ptr = ReadInt32 (m_start, offset);
        }

        // Advance the offset
        *offset_ptr = offset;
        // Return a non-NULL pointer to the converted data as an indicator of success
        return dst;
    }
    return NULL;
}

//----------------------------------------------------------------------
// Extract a single uint64_t from the data and update the offset
// pointed to by "offset_ptr".
//
// RETURNS the uint64_t that was extracted, or zero on failure.
//----------------------------------------------------------------------
uint64_t
DataExtractor::GetU64 (uint32_t *offset_ptr) const
{
    uint64_t val = 0;
    register uint32_t offset = *offset_ptr;
    if ( ValidOffsetForDataOfSize(offset, sizeof(val)) )
    {
        if (m_byte_order != lldb::endian::InlHostByteOrder())
            val = ReadSwapInt64 (m_start, offset);
        else
            val = ReadInt64 (m_start, offset);

        // Advance the offset
        *offset_ptr += sizeof(val);
    }
    return val;
}

//----------------------------------------------------------------------
// GetU64
//
// Get multiple consecutive 64 bit values. Return true if the entire
// read succeeds and increment the offset pointed to by offset_ptr, else
// return false and leave the offset pointed to by offset_ptr unchanged.
//----------------------------------------------------------------------
void *
DataExtractor::GetU64 (uint32_t *offset_ptr, void *void_dst, uint32_t count) const
{
    uint64_t *dst = (uint64_t *)void_dst;
    const size_t value_size = sizeof(uint64_t);
    register uint32_t offset = *offset_ptr;

    if ((count > 0) && ValidOffsetForDataOfSize(offset, value_size * count))
    {
        uint64_t *value_ptr;
        uint64_t *end = dst + count;
        if (m_byte_order != lldb::endian::InlHostByteOrder())
        {
            for (value_ptr = dst; value_ptr < end; ++value_ptr, offset += value_size)
                *value_ptr = ReadSwapInt64 (m_start, offset);

        }
        else
        {
            for (value_ptr = dst; value_ptr < end; ++value_ptr, offset += value_size)
                *value_ptr = ReadInt64 (m_start, offset);
        }

        // Advance the offset
        *offset_ptr = offset;
        // Return a non-NULL pointer to the converted data as an indicator of success
        return dst;
    }
    return NULL;
}

//----------------------------------------------------------------------
// Extract a single integer value from the data and update the offset
// pointed to by "offset_ptr". The size of the extracted integer
// is specified by the "byte_size" argument. "byte_size" should have
// a value between 1 and 4 since the return value is only 32 bits
// wide. Any "byte_size" values less than 1 or greater than 4 will
// result in nothing being extracted, and zero being returned.
//
// RETURNS the integer value that was extracted, or zero on failure.
//----------------------------------------------------------------------
uint32_t
DataExtractor::GetMaxU32 (uint32_t *offset_ptr, uint32_t byte_size) const
{
    switch (byte_size)
    {
    case 1: return GetU8 (offset_ptr); break;
    case 2: return GetU16(offset_ptr); break;
    case 4: return GetU32(offset_ptr); break;
    default:
        assert(!"GetMaxU32 unhandled case!");
        break;
    }
    return 0;
}

//----------------------------------------------------------------------
// Extract a single integer value from the data and update the offset
// pointed to by "offset_ptr". The size of the extracted integer
// is specified by the "byte_size" argument. "byte_size" should have
// a value >= 1 and <= 8 since the return value is only 64 bits
// wide. Any "byte_size" values less than 1 or greater than 8 will
// result in nothing being extracted, and zero being returned.
//
// RETURNS the integer value that was extracted, or zero on failure.
//----------------------------------------------------------------------
uint64_t
DataExtractor::GetMaxU64 (uint32_t *offset_ptr, uint32_t size) const
{
    switch (size)
    {
    case 1: return GetU8 (offset_ptr); break;
    case 2: return GetU16(offset_ptr); break;
    case 4: return GetU32(offset_ptr); break;
    case 8: return GetU64(offset_ptr); break;
    default:
        assert(!"GetMax64 unhandled case!");
        break;
    }
    return 0;
}

uint64_t
DataExtractor::GetMaxU64_unchecked (uint32_t *offset_ptr, uint32_t size) const
{
    switch (size)
    {
        case 1: return GetU8_unchecked  (offset_ptr); break;
        case 2: return GetU16_unchecked (offset_ptr); break;
        case 4: return GetU32_unchecked (offset_ptr); break;
        case 8: return GetU64_unchecked (offset_ptr); break;
        default:
            assert(!"GetMax64 unhandled case!");
            break;
    }
    return 0;
}

int64_t
DataExtractor::GetMaxS64 (uint32_t *offset_ptr, uint32_t size) const
{
    switch (size)
    {
    case 1: return (int8_t)GetU8 (offset_ptr); break;
    case 2: return (int16_t)GetU16(offset_ptr); break;
    case 4: return (int32_t)GetU32(offset_ptr); break;
    case 8: return (int64_t)GetU64(offset_ptr); break;
    default:
        assert(!"GetMax64 unhandled case!");
        break;
    }
    return 0;
}

uint64_t
DataExtractor::GetMaxU64Bitfield (uint32_t *offset_ptr, uint32_t size, uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset) const
{
    uint64_t uval64 = GetMaxU64 (offset_ptr, size);
    if (bitfield_bit_size > 0)
    {
        if (bitfield_bit_offset > 0)
            uval64 >>= bitfield_bit_offset;
        uint64_t bitfield_mask = ((1ul << bitfield_bit_size) - 1);
        if (!bitfield_mask && bitfield_bit_offset == 0 && bitfield_bit_size == 64)
            return uval64;
        uval64 &= bitfield_mask;
    }
    return uval64;
}

int64_t
DataExtractor::GetMaxS64Bitfield (uint32_t *offset_ptr, uint32_t size, uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset) const
{
    int64_t sval64 = GetMaxS64 (offset_ptr, size);
    if (bitfield_bit_size > 0)
    {
        if (bitfield_bit_offset > 0)
            sval64 >>= bitfield_bit_offset;
        uint64_t bitfield_mask = (((uint64_t)1) << bitfield_bit_size) - 1;
        sval64 &= bitfield_mask;
        // sign extend if needed
        if (sval64 & (((uint64_t)1) << (bitfield_bit_size - 1)))
            sval64 |= ~bitfield_mask;
    }
    return sval64;
}


float
DataExtractor::GetFloat (uint32_t *offset_ptr) const
{
    typedef float float_type;
    float_type val = 0.0;
    const uint8_t *src_data = PeekData (*offset_ptr, sizeof(float_type));
    
    if (src_data)
    {
        if (m_byte_order != lldb::endian::InlHostByteOrder())
        {
            uint8_t *dst_data = (uint8_t *)&val;
            for (int i=0; i<sizeof(float_type); ++i)
                dst_data[sizeof(float_type) - 1 - i] = src_data[i];
        }
        else
        {
            ::memcpy (&val, src_data, sizeof (float_type));
        }

        // Advance the offset
        *offset_ptr += sizeof(val);
    }
    return val;
}

double
DataExtractor::GetDouble (uint32_t *offset_ptr) const
{
    typedef double float_type;
    float_type val = 0.0;
    const uint8_t *src_data = PeekData (*offset_ptr, sizeof(float_type));
    
    if (src_data)
    {
        if (m_byte_order != lldb::endian::InlHostByteOrder())
        {
            uint8_t *dst_data = (uint8_t *)&val;
            for (int i=0; i<sizeof(float_type); ++i)
                dst_data[sizeof(float_type) - 1 - i] = src_data[i];
        }
        else
        {
            ::memcpy (&val, src_data, sizeof (float_type));
        }

        // Advance the offset
        *offset_ptr += sizeof(val);
    }
    return val;
}


long double
DataExtractor::GetLongDouble (uint32_t *offset_ptr) const
{
    typedef long double float_type;
    float_type val = 0.0;
    const uint8_t *src_data = PeekData (*offset_ptr, sizeof(float_type));
    
    if (src_data)
    {
        if (m_byte_order != lldb::endian::InlHostByteOrder())
        {
            uint8_t *dst_data = (uint8_t *)&val;
            for (int i=0; i<sizeof(float_type); ++i)
                dst_data[sizeof(float_type) - 1 - i] = src_data[i];
        }
        else
        {
            ::memcpy (&val, src_data, sizeof (float_type));
        }

        // Advance the offset
        *offset_ptr += sizeof(val);
    }
    return val;
}


//------------------------------------------------------------------
// Extract a single address from the data and update the offset
// pointed to by "offset_ptr". The size of the extracted address
// comes from the "this->m_addr_size" member variable and should be
// set correctly prior to extracting any address values.
//
// RETURNS the address that was extracted, or zero on failure.
//------------------------------------------------------------------
uint64_t
DataExtractor::GetAddress (uint32_t *offset_ptr) const
{
    return GetMaxU64 (offset_ptr, m_addr_size);
}

uint64_t
DataExtractor::GetAddress_unchecked (uint32_t *offset_ptr) const
{
    return GetMaxU64_unchecked (offset_ptr, m_addr_size);
}

//------------------------------------------------------------------
// Extract a single pointer from the data and update the offset
// pointed to by "offset_ptr". The size of the extracted pointer
// comes from the "this->m_addr_size" member variable and should be
// set correctly prior to extracting any pointer values.
//
// RETURNS the pointer that was extracted, or zero on failure.
//------------------------------------------------------------------
uint64_t
DataExtractor::GetPointer (uint32_t *offset_ptr) const
{
    return GetMaxU64 (offset_ptr, m_addr_size);
}

//----------------------------------------------------------------------
// GetDwarfEHPtr
//
// Used for calls when the value type is specified by a DWARF EH Frame
// pointer encoding.
//----------------------------------------------------------------------

uint64_t
DataExtractor::GetGNUEHPointer (uint32_t *offset_ptr, uint32_t eh_ptr_enc, lldb::addr_t pc_rel_addr, lldb::addr_t text_addr, lldb::addr_t data_addr)//, BSDRelocs *data_relocs) const
{
    if (eh_ptr_enc == DW_EH_PE_omit)
        return ULLONG_MAX;  // Value isn't in the buffer...

    uint64_t baseAddress = 0;
    uint64_t addressValue = 0;
    const uint32_t addr_size = GetAddressByteSize();

    bool signExtendValue = false;
    // Decode the base part or adjust our offset
    switch (eh_ptr_enc & 0x70)
    {
    case DW_EH_PE_pcrel:
        signExtendValue = true;
        baseAddress = *offset_ptr;
        if (pc_rel_addr != LLDB_INVALID_ADDRESS)
            baseAddress += pc_rel_addr;
//      else
//          Log::GlobalWarning ("PC relative pointer encoding found with invalid pc relative address.");
        break;

    case DW_EH_PE_textrel:
        signExtendValue = true;
        if (text_addr != LLDB_INVALID_ADDRESS)
            baseAddress = text_addr;
//      else
//          Log::GlobalWarning ("text relative pointer encoding being decoded with invalid text section address, setting base address to zero.");
        break;

    case DW_EH_PE_datarel:
        signExtendValue = true;
        if (data_addr != LLDB_INVALID_ADDRESS)
            baseAddress = data_addr;
//      else
//          Log::GlobalWarning ("data relative pointer encoding being decoded with invalid data section address, setting base address to zero.");
        break;

    case DW_EH_PE_funcrel:
        signExtendValue = true;
        break;

    case DW_EH_PE_aligned:
        {
            // SetPointerSize should be called prior to extracting these so the
            // pointer size is cached
            assert(addr_size != 0);
            if (addr_size)
            {
                // Align to a address size boundary first
                uint32_t alignOffset = *offset_ptr % addr_size;
                if (alignOffset)
                    offset_ptr += addr_size - alignOffset;
            }
        }
        break;

    default:
    break;
    }

    // Decode the value part
    switch (eh_ptr_enc & DW_EH_PE_MASK_ENCODING)
    {
    case DW_EH_PE_absptr    :
        {
            addressValue = GetAddress (offset_ptr);
//          if (data_relocs)
//              addressValue = data_relocs->Relocate(*offset_ptr - addr_size, *this, addressValue);
        }
        break;
    case DW_EH_PE_uleb128   : addressValue = GetULEB128(offset_ptr);        break;
    case DW_EH_PE_udata2    : addressValue = GetU16(offset_ptr);            break;
    case DW_EH_PE_udata4    : addressValue = GetU32(offset_ptr);            break;
    case DW_EH_PE_udata8    : addressValue = GetU64(offset_ptr);            break;
    case DW_EH_PE_sleb128   : addressValue = GetSLEB128(offset_ptr);        break;
    case DW_EH_PE_sdata2    : addressValue = (int16_t)GetU16(offset_ptr);   break;
    case DW_EH_PE_sdata4    : addressValue = (int32_t)GetU32(offset_ptr);   break;
    case DW_EH_PE_sdata8    : addressValue = (int64_t)GetU64(offset_ptr);   break;
    default:
    // Unhandled encoding type
    assert(eh_ptr_enc);
    break;
    }

    // Since we promote everything to 64 bit, we may need to sign extend
    if (signExtendValue && addr_size < sizeof(baseAddress))
    {
        uint64_t sign_bit = 1ull << ((addr_size * 8ull) - 1ull);
        if (sign_bit & addressValue)
        {
            uint64_t mask = ~sign_bit + 1;
            addressValue |= mask;
        }
    }
    return baseAddress + addressValue;
}

size_t
DataExtractor::ExtractBytes (uint32_t offset, uint32_t length, ByteOrder dst_byte_order, void *dst) const
{
    const uint8_t *src = PeekData (offset, length);
    if (src)
    {
        if (dst_byte_order != GetByteOrder())
        {
            for (uint32_t i=0; i<length; ++i)
                ((uint8_t*)dst)[i] = src[length - i - 1];
        }
        else
            ::memcpy (dst, src, length);
        return length;
    }
    return 0;
}
//----------------------------------------------------------------------
// Peeks at bytes in the contained data.
//
// Returns a valid pointer to bytes if "offset" is a valid offset in
// and there are "length" bytes available, else NULL is returned.
//----------------------------------------------------------------------
const uint8_t*
DataExtractor::PeekData (uint32_t offset, uint32_t length) const
{
    if ( length > 0 && ValidOffsetForDataOfSize(offset, length) )
        return m_start + offset;
    return NULL;
}

//----------------------------------------------------------------------
// Returns a pointer to a bytes in this object's data at the offset
// pointed to by "offset_ptr". If "length" is zero or too large,
// then the offset pointed to by "offset_ptr" will not be updated
// and NULL will be returned.
//
// Returns a pointer to the data if the offset and length are valid,
// or NULL otherwise.
//----------------------------------------------------------------------
const void*
DataExtractor::GetData (uint32_t *offset_ptr, uint32_t length) const
{
    const uint8_t* bytes = NULL;
    register uint32_t offset = *offset_ptr;
    if ( length > 0 && ValidOffsetForDataOfSize(offset, length) )
    {
        bytes = m_start + offset;
        *offset_ptr = offset + length;
    }
    return bytes;
}

// Extract data and swap if needed when doing the copy
uint32_t
DataExtractor::CopyByteOrderedData (uint32_t src_offset, 
                                    uint32_t src_len,
                                    void *dst_void_ptr, 
                                    uint32_t dst_len, 
                                    ByteOrder dst_byte_order) const
{
    // Validate the source info
    assert (ValidOffsetForDataOfSize(src_offset, src_len));
    assert (src_len > 0);
    assert (m_byte_order == eByteOrderBig || m_byte_order == eByteOrderLittle);

    // Validate the destination info
    assert (dst_void_ptr != NULL);
    assert (dst_len > 0);
    assert (dst_byte_order == eByteOrderBig || dst_byte_order == eByteOrderLittle);
    
    // Must have valid byte orders set in this object and for destination
    if (!(dst_byte_order == eByteOrderBig || dst_byte_order == eByteOrderLittle) ||
        !(m_byte_order == eByteOrderBig || m_byte_order == eByteOrderLittle))
        return 0;

    uint32_t i;
    uint8_t* dst = (uint8_t*)dst_void_ptr;
    const uint8_t* src = (const uint8_t *)PeekData (src_offset, src_len);
    if (src)
    {
        if (dst_len >= src_len)
        {
            // We are copying the entire value from src into dst.
            // Calculate how many, if any, zeroes we need for the most 
            // significant bytes if "dst_len" is greater than "src_len"...
            const uint32_t num_zeroes = dst_len - src_len;
            if (dst_byte_order == eByteOrderBig)
            {
                // Big endian, so we lead with zeroes...
                if (num_zeroes > 0)
                    ::memset (dst, 0, num_zeroes);
                // Then either copy or swap the rest
                if (m_byte_order == eByteOrderBig)
                {
                    ::memcpy (dst + num_zeroes, src, src_len);
                }
                else
                {
                    for (i=0; i<src_len; ++i)
                        dst[i+num_zeroes] = src[src_len - 1 - i];
                }
            }
            else
            {
                // Little endian destination, so we lead the value bytes
                if (m_byte_order == eByteOrderBig)
                {
                    for (i=0; i<src_len; ++i)
                        dst[i] = src[src_len - 1 - i];
                }
                else
                {
                    ::memcpy (dst, src, src_len);
                }
                // And zero the rest...
                if (num_zeroes > 0)
                    ::memset (dst + src_len, 0, num_zeroes);
            }
            return src_len;
        }
        else
        {
            // We are only copying some of the value from src into dst..

            if (dst_byte_order == eByteOrderBig)
            {
                // Big endian dst
                if (m_byte_order == eByteOrderBig)
                {
                    // Big endian dst, with big endian src
                    ::memcpy (dst, src + (src_len - dst_len), dst_len);
                }
                else
                {
                    // Big endian dst, with little endian src
                    for (i=0; i<dst_len; ++i)
                        dst[i] = src[dst_len - 1 - i];
                }
            }
            else
            {
                // Little endian dst
                if (m_byte_order == eByteOrderBig)
                {
                    // Little endian dst, with big endian src
                    for (i=0; i<dst_len; ++i)
                        dst[i] = src[src_len - 1 - i];
                }
                else
                {
                    // Little endian dst, with big endian src
                    ::memcpy (dst, src, dst_len);
                }
            }
            return dst_len;
        }            

    }
    return 0;
}


//----------------------------------------------------------------------
// Extracts a AsCString (fixed length, or variable length) from
// the data at the offset pointed to by "offset_ptr". If
// "length" is zero, then a variable length NULL terminated C
// string will be extracted from the data the "offset_ptr" will be
// updated with the offset of the byte that follows the NULL
// terminator byte. If "length" is greater than zero, then
// the function will make sure there are "length" bytes
// available in the current data and if so, return a valid pointer.
//
// If the offset pointed to by "offset_ptr" is out of bounds, or if
// "length" is non-zero and there aren't enough avaialable
// bytes, NULL will be returned and "offset_ptr" will not be
// updated.
//----------------------------------------------------------------------
const char*
DataExtractor::GetCStr (uint32_t *offset_ptr) const
{
    const char *s = NULL;
    if ( m_start < m_end )
    {
        s = (char*)m_start + *offset_ptr;

        size_t length = strlen(s) + 1;

        if (!ValidOffsetForDataOfSize(*offset_ptr, length))
            return NULL;

        // Advance the offset
        *offset_ptr += length;
    }
    return s;
}

//------------------------------------------------------------------
// Peeks at a string in the contained data. No verification is done
// to make sure the entire string lies within the bounds of this
// object's data, only "offset" is verified to be a valid offset.
//
// Returns a valid C string pointer if "offset" is a valid offset in
// this object's data, else NULL is returned.
//------------------------------------------------------------------
const char *
DataExtractor::PeekCStr (uint32_t offset) const
{
    if (ValidOffset (offset))
        return (const char*)m_start + offset;
    return NULL;
}

//----------------------------------------------------------------------
// Extracts an unsigned LEB128 number from this object's data
// starting at the offset pointed to by "offset_ptr". The offset
// pointed to by "offset_ptr" will be updated with the offset of the
// byte following the last extracted byte.
//
// Returned the extracted integer value.
//----------------------------------------------------------------------
uint64_t
DataExtractor::GetULEB128 (uint32_t *offset_ptr) const
{
    const uint8_t *src = m_start + *offset_ptr;
    const uint8_t *end = m_end;
    
    if (src < end)
    {
        uint64_t result = *src++;
        if (result >= 0x80)
        {
            result &= 0x7f;
            int shift = 7;
            while (src < end)
            {
                uint8_t byte = *src++;
                result |= (byte & 0x7f) << shift;
                if ((byte & 0x80) == 0)
                    break;
                shift += 7;
            }
        }
        *offset_ptr = (uint32_t)(src - m_start);
        return result;
    }
    
    return 0;
}

//----------------------------------------------------------------------
// Extracts an signed LEB128 number from this object's data
// starting at the offset pointed to by "offset_ptr". The offset
// pointed to by "offset_ptr" will be updated with the offset of the
// byte following the last extracted byte.
//
// Returned the extracted integer value.
//----------------------------------------------------------------------
int64_t
DataExtractor::GetSLEB128 (uint32_t *offset_ptr) const
{
    int64_t result = 0;

    if ( m_start < m_end )
    {
        int shift = 0;
        int size = sizeof (uint32_t) * 8;
        const uint8_t *src = m_start + *offset_ptr;

        uint8_t byte = 0;
        int bytecount = 0;

        while (src < m_end)
        {
            bytecount++;
            byte = *src++;
            result |= (byte & 0x7f) << shift;
            shift += 7;
            if ((byte & 0x80) == 0)
                break;
        }

        // Sign bit of byte is 2nd high order bit (0x40)
        if (shift < size && (byte & 0x40))
            result |= - (1 << shift);

        *offset_ptr += bytecount;
    }
    return result;
}

//----------------------------------------------------------------------
// Skips a ULEB128 number (signed or unsigned) from this object's
// data starting at the offset pointed to by "offset_ptr". The
// offset pointed to by "offset_ptr" will be updated with the offset
// of the byte following the last extracted byte.
//
// Returns the number of bytes consumed during the extraction.
//----------------------------------------------------------------------
uint32_t
DataExtractor::Skip_LEB128 (uint32_t *offset_ptr) const
{
    uint32_t bytes_consumed = 0;
    if ( m_start < m_end )
    {
        const uint8_t *start = m_start + *offset_ptr;
        const uint8_t *src = start;

        while ((src < m_end) && (*src++ & 0x80))
            ++bytes_consumed;

        *offset_ptr += src - start;
    }
    return bytes_consumed;
}

static uint32_t
DumpAPInt (Stream *s, const DataExtractor &data, uint32_t offset, uint32_t byte_size, bool is_signed, unsigned radix)
{
    llvm::SmallVector<uint64_t, 2> uint64_array;
    uint32_t bytes_left = byte_size;
    uint64_t u64;
    const lldb::ByteOrder byte_order = data.GetByteOrder();
    if (byte_order == lldb::eByteOrderLittle)
    {
        while (bytes_left > 0)
        {
            if (bytes_left >= 8)
            {
                u64 = data.GetU64(&offset);
                bytes_left -= 8;
            }
            else
            {
                u64 = data.GetMaxU64(&offset, bytes_left);
                bytes_left = 0;
            }                        
            uint64_array.push_back(u64);
        }
    }
    else if (byte_order == lldb::eByteOrderBig)
    {
        uint32_t be_offset = offset + byte_size;
        uint32_t temp_offset;
        while (bytes_left > 0)
        {
            if (bytes_left >= 8)
            {
                be_offset -= 8;
                temp_offset = be_offset;
                u64 = data.GetU64(&temp_offset);
                bytes_left -= 8;
            }
            else
            {
                be_offset -= bytes_left;
                temp_offset = be_offset;
                u64 = data.GetMaxU64(&temp_offset, bytes_left);
                bytes_left = 0;
            }                        
            uint64_array.push_back(u64);
        }
    }
    else
        return offset;

    llvm::APInt apint (byte_size * 8, llvm::ArrayRef<uint64_t>(uint64_array));
 
    std::string apint_str(apint.toString(radix, is_signed));
    switch (radix)
    {
        case 2:
            s->Write ("0b", 2);
            break;
        case 8:
            s->Write ("0", 1);
            break;
        case 10:
            break;
    }
    s->Write(apint_str.c_str(), apint_str.size());
    return offset;
}

uint32_t
DataExtractor::Dump (Stream *s,
                     uint32_t start_offset,
                     lldb::Format item_format,
                     uint32_t item_byte_size,
                     uint32_t item_count,
                     uint32_t num_per_line,
                     uint64_t base_addr,
                     uint32_t item_bit_size,     // If zero, this is not a bitfield value, if non-zero, the value is a bitfield
                     uint32_t item_bit_offset,    // If "item_bit_size" is non-zero, this is the shift amount to apply to a bitfield
                     ExecutionContextScope *exe_scope) const
{
    if (s == NULL)
        return start_offset;

    if (item_format == eFormatPointer)
    {
        if (item_byte_size != 4 && item_byte_size != 8)
            item_byte_size = s->GetAddressByteSize();
    }
    
    uint32_t offset = start_offset;

    if (item_format == eFormatInstruction)
    {
        TargetSP target_sp;
        if (exe_scope)
            target_sp = exe_scope->CalculateTarget();
        if (target_sp)
        {
            DisassemblerSP disassembler_sp (Disassembler::FindPlugin(target_sp->GetArchitecture(), NULL));
            if (disassembler_sp)
            {
                lldb::addr_t addr = base_addr + start_offset;
                lldb_private::Address so_addr;
                if (!target_sp->GetSectionLoadList().ResolveLoadAddress(addr, so_addr))
                {
                    so_addr.SetRawAddress(addr);
                }

                size_t bytes_consumed = disassembler_sp->DecodeInstructions (so_addr, *this, start_offset, item_count, false);
                
                if (bytes_consumed)
                {
                    offset += bytes_consumed;
                    const bool show_address = base_addr != LLDB_INVALID_ADDRESS;
                    const bool show_bytes = true;
                    ExecutionContext exe_ctx;
                    exe_scope->CalculateExecutionContext(exe_ctx);
                    disassembler_sp->GetInstructionList().Dump (s,  show_address, show_bytes, &exe_ctx);
                }
            }
        }
        else
            s->Printf ("invalid target");

        return offset;
    }

    if ((item_format == eFormatOSType || item_format == eFormatAddressInfo) && item_byte_size > 8)
        item_format = eFormatHex;

    uint32_t line_start_offset = start_offset;
    for (uint32_t count = 0; ValidOffset(offset) && count < item_count; ++count)
    {
        if ((count % num_per_line) == 0)
        {
            if (count > 0)
            {
                if (item_format == eFormatBytesWithASCII && offset > line_start_offset)
                {
                    s->Printf("%*s", (num_per_line - (offset - line_start_offset)) * 3 + 2, "");
                    Dump(s, line_start_offset, eFormatCharPrintable, 1, offset - line_start_offset, UINT32_MAX, LLDB_INVALID_ADDRESS, 0, 0);
                }
                s->EOL();
            }
            if (base_addr != LLDB_INVALID_ADDRESS)
                s->Printf ("0x%8.8llx: ", (uint64_t)(base_addr + (offset - start_offset)));
            line_start_offset = offset;
        }
        else
        if (item_format != eFormatChar &&
            item_format != eFormatCharPrintable &&
            item_format != eFormatCharArray &&
            count > 0)
        {
            s->PutChar(' ');
        }

        uint32_t i;
        switch (item_format)
        {
        case eFormatBoolean:
            if (item_byte_size <= 8)
                s->Printf ("%s", GetMaxU64Bitfield(&offset, item_byte_size, item_bit_size, item_bit_offset) ? "true" : "false");
            else
            {
                s->Printf("error: unsupported byte size (%u) for boolean format", item_byte_size);
                return offset;
            }
            break;

        case eFormatBinary:
            if (item_byte_size <= 8)
            {
                uint64_t uval64 = GetMaxU64Bitfield(&offset, item_byte_size, item_bit_size, item_bit_offset);
                // Avoid std::bitset<64>::to_string() since it is missing in
                // earlier C++ libraries
                std::string binary_value(64, '0');
                std::bitset<64> bits(uval64);
                for (i = 0; i < 64; ++i)
                    if (bits[i])
                        binary_value[64 - 1 - i] = '1';
                if (item_bit_size > 0)
                    s->Printf("0b%s", binary_value.c_str() + 64 - item_bit_size);
                else if (item_byte_size > 0 && item_byte_size <= 8)
                    s->Printf("0b%s", binary_value.c_str() + 64 - item_byte_size * 8);
            }
            else
            {
                const bool is_signed = false;
                const unsigned radix = 2;
                offset = DumpAPInt (s, *this, offset, item_byte_size, is_signed, radix);
            }
            break;

        case eFormatBytes:
        case eFormatBytesWithASCII:
            for (i=0; i<item_byte_size; ++i)
            {
                s->Printf ("%2.2x", GetU8(&offset));
            }
            // Put an extra space between the groups of bytes if more than one
            // is being dumped in a group (item_byte_size is more than 1).
            if (item_byte_size > 1)
                s->PutChar(' ');
            break;

        case eFormatChar:
        case eFormatCharPrintable:
        case eFormatCharArray:
            {
                // If we are only printing one character surround it with single
                // quotes
                if (item_count == 1 && item_format == eFormatChar)
                    s->PutChar('\'');

                const uint64_t ch = GetMaxU64Bitfield(&offset, item_byte_size, item_bit_size, item_bit_offset);
                if (isprint(ch))
                    s->Printf ("%c", (char)ch);
                else if (item_format != eFormatCharPrintable)
                {
                    switch (ch)
                    {
                    case '\033': s->Printf ("\\e"); break;
                    case '\a': s->Printf ("\\a"); break;
                    case '\b': s->Printf ("\\b"); break;
                    case '\f': s->Printf ("\\f"); break;
                    case '\n': s->Printf ("\\n"); break;
                    case '\r': s->Printf ("\\r"); break;
                    case '\t': s->Printf ("\\t"); break;
                    case '\v': s->Printf ("\\v"); break;
                    case '\0': s->Printf ("\\0"); break;
                    default:   
                        if (item_byte_size == 1)
                            s->Printf ("\\x%2.2x", (uint8_t)ch); 
                        else
                            s->Printf ("%llu", ch); 
                        break;
                    }
                }
                else
                {
                    s->PutChar(NON_PRINTABLE_CHAR);
                }

                // If we are only printing one character surround it with single quotes
                if (item_count == 1 && item_format == eFormatChar)
                    s->PutChar('\'');
            }
            break;

        case eFormatEnum:       // Print enum value as a signed integer when we don't get the enum type
        case eFormatDecimal:
            if (item_byte_size <= 8)
                s->Printf ("%lld", GetMaxS64Bitfield(&offset, item_byte_size, item_bit_size, item_bit_offset));
            else
            {
                const bool is_signed = true;
                const unsigned radix = 10;
                offset = DumpAPInt (s, *this, offset, item_byte_size, is_signed, radix);
            }
            break;

        case eFormatUnsigned:
            if (item_byte_size <= 8)
                s->Printf ("%llu", GetMaxU64Bitfield(&offset, item_byte_size, item_bit_size, item_bit_offset));
            else
            {
                const bool is_signed = false;
                const unsigned radix = 10;
                offset = DumpAPInt (s, *this, offset, item_byte_size, is_signed, radix);
            }
            break;

        case eFormatOctal:
            if (item_byte_size <= 8)
                s->Printf ("0%llo", GetMaxS64Bitfield(&offset, item_byte_size, item_bit_size, item_bit_offset));
            else
            {
                const bool is_signed = false;
                const unsigned radix = 8;
                offset = DumpAPInt (s, *this, offset, item_byte_size, is_signed, radix);
            }
            break;

        case eFormatOSType:
            {
                uint64_t uval64 = GetMaxU64Bitfield(&offset, item_byte_size, item_bit_size, item_bit_offset);
                s->PutChar('\'');
                for (i=0; i<item_byte_size; ++i)
                {
                    uint8_t ch = (uint8_t)(uval64 >> ((item_byte_size - i - 1) * 8));
                    if (isprint(ch))
                        s->Printf ("%c", ch);
                    else
                    {
                        switch (ch)
                        {
                        case '\033': s->Printf ("\\e"); break;
                        case '\a': s->Printf ("\\a"); break;
                        case '\b': s->Printf ("\\b"); break;
                        case '\f': s->Printf ("\\f"); break;
                        case '\n': s->Printf ("\\n"); break;
                        case '\r': s->Printf ("\\r"); break;
                        case '\t': s->Printf ("\\t"); break;
                        case '\v': s->Printf ("\\v"); break;
                        case '\0': s->Printf ("\\0"); break;
                        default:   s->Printf ("\\x%2.2x", ch); break;
                        }
                    }
                }
                s->PutChar('\'');
            }
            break;
            
        case eFormatCString:
            {
                const char *cstr = GetCStr(&offset);
                
                if (!cstr)
                {
                    s->Printf("NULL");
                    offset = UINT32_MAX;
                }
                else
                {
                    s->PutChar('\"');
                    
                    while (const char c = *cstr)
                    {                    
                        if (isprint(c))
                        {
                            s->PutChar(c);
                        }
                        else
                        {
                            switch (c)
                            {
                            case '\033': s->Printf ("\\e"); break;
                            case '\a': s->Printf ("\\a"); break;
                            case '\b': s->Printf ("\\b"); break;
                            case '\f': s->Printf ("\\f"); break;
                            case '\n': s->Printf ("\\n"); break;
                            case '\r': s->Printf ("\\r"); break;
                            case '\t': s->Printf ("\\t"); break;
                            case '\v': s->Printf ("\\v"); break;
                            default:   s->Printf ("\\x%2.2x", c); break;
                            }
                        }
                        
                        ++cstr;
                    }
                    
                    s->PutChar('\"');
                }
            }
            break;


        case eFormatPointer:
            s->Address(GetMaxU64Bitfield(&offset, item_byte_size, item_bit_size, item_bit_offset), sizeof (addr_t));
            break;


        case eFormatComplexInteger:
            {
                uint32_t complex_int_byte_size = item_byte_size / 2;
                
                if (complex_int_byte_size <= 8)
                {
                    s->Printf("%llu", GetMaxU64Bitfield(&offset, complex_int_byte_size, 0, 0));
                    s->Printf(" + %llui", GetMaxU64Bitfield(&offset, complex_int_byte_size, 0, 0));
                }
                else
                {
                    s->Printf("error: unsupported byte size (%u) for complex integer format", item_byte_size);
                    return offset;
                }
            }
            break;

        case eFormatComplex:
            if (sizeof(float) * 2 == item_byte_size)
            {
                float f32_1 = GetFloat (&offset);
                float f32_2 = GetFloat (&offset);

                s->Printf ("%g + %gi", f32_1, f32_2);
                break;
            }
            else if (sizeof(double) * 2 == item_byte_size)
            {
                double d64_1 = GetDouble (&offset);
                double d64_2 = GetDouble (&offset);

                s->Printf ("%lg + %lgi", d64_1, d64_2);
                break;
            }
            else if (sizeof(long double) * 2 == item_byte_size)
            {
                long double ld64_1 = GetLongDouble (&offset);
                long double ld64_2 = GetLongDouble (&offset);
                s->Printf ("%Lg + %Lgi", ld64_1, ld64_2);
                break;
            }
            else
            {
                s->Printf("error: unsupported byte size (%u) for complex float format", item_byte_size);
                return offset;
            }
            break;

        default:
        case eFormatDefault:
        case eFormatHex:
        case eFormatHexUppercase:
            {
                bool wantsuppercase  = (item_format == eFormatHexUppercase);
                if (item_byte_size <= 8)
                {
                    s->Printf(wantsuppercase ? "0x%*.*llX" : "0x%*.*llx", 2 * item_byte_size, 2 * item_byte_size, GetMaxU64Bitfield(&offset, item_byte_size, item_bit_size, item_bit_offset));
                }
                else
                {
                    assert (item_bit_size == 0 && item_bit_offset == 0);
                    s->PutCString("0x");
                    const uint8_t *bytes = (const uint8_t* )GetData(&offset, item_byte_size);
                    if (bytes)
                    {
                        uint32_t idx;
                        if (m_byte_order == eByteOrderBig)
                        {
                            for (idx = 0; idx < item_byte_size; ++idx)
                                s->Printf(wantsuppercase ? "%2.2X" : "%2.2x", bytes[idx]);
                        }
                        else
                        {
                            for (idx = 0; idx < item_byte_size; ++idx)
                                s->Printf(wantsuppercase ? "%2.2X" : "%2.2x", bytes[item_byte_size - 1 - idx]);
                        }
                    }
                }
            }
            break;

        case eFormatFloat:
            {
                std::ostringstream ss;
                if (item_byte_size == sizeof(float))
                {
                    ss.precision(std::numeric_limits<float>::digits10);
                    ss << GetFloat(&offset);
                } 
                else if (item_byte_size == sizeof(double))
                {
                    ss.precision(std::numeric_limits<double>::digits10);
                    ss << GetDouble(&offset);
                }
                else if (item_byte_size == sizeof(long double))
                {
                    ss.precision(std::numeric_limits<long double>::digits10);
                    ss << GetLongDouble(&offset);
                }
                else
                {
                    s->Printf("error: unsupported byte size (%u) for float format", item_byte_size);
                    return offset;
                }
                ss.flush();
                s->Printf("%s", ss.str().c_str());
            }
            break;

        case eFormatUnicode16:
            s->Printf("U+%4.4x", GetU16 (&offset));
            break;

        case eFormatUnicode32:
            s->Printf("U+0x%8.8x", GetU32 (&offset));
            break;

        case eFormatAddressInfo:
            {
                addr_t addr = GetMaxU64Bitfield(&offset, item_byte_size, item_bit_size, item_bit_offset);
                s->Printf("0x%*.*llx", 2 * item_byte_size, 2 * item_byte_size, addr);
                if (exe_scope)
                {
                    TargetSP target_sp (exe_scope->CalculateTarget());
                    lldb_private::Address so_addr;
                    if (target_sp)
                    {
                        if (target_sp->GetSectionLoadList().ResolveLoadAddress(addr, so_addr))
                        {
                            s->PutChar(' ');
                            so_addr.Dump (s,
                                          exe_scope,
                                          Address::DumpStyleResolvedDescription,
                                          Address::DumpStyleModuleWithFileAddress);
                        }
                        else
                        {
                            so_addr.SetOffset(addr);
                            so_addr.Dump (s, exe_scope, Address::DumpStyleResolvedPointerDescription);
                        }
                    }
                }
            }
            break;

        case eFormatHexFloat:
            if (sizeof(float) == item_byte_size)
            {
                char float_cstr[256];
                llvm::APFloat ap_float (GetFloat (&offset));
                ap_float.convertToHexString (float_cstr, 0, false, llvm::APFloat::rmNearestTiesToEven);
                s->Printf ("%s", float_cstr);
                break;
            }
            else if (sizeof(double) == item_byte_size)
            {
                char float_cstr[256];
                llvm::APFloat ap_float (GetDouble (&offset));
                ap_float.convertToHexString (float_cstr, 0, false, llvm::APFloat::rmNearestTiesToEven);
                s->Printf ("%s", float_cstr);
                break;
            }
            else
            {
                s->Printf("error: unsupported byte size (%u) for hex float format", item_byte_size);
                return offset;
            }
            break;

// please keep the single-item formats below in sync with FormatManager::GetSingleItemFormat
// if you fail to do so, users will start getting different outputs depending on internal
// implementation details they should not care about ||
        case eFormatVectorOfChar:               //   ||
            s->PutChar('{');                    //   \/   
            offset = Dump (s, offset, eFormatCharArray, 1, item_byte_size, item_byte_size, LLDB_INVALID_ADDRESS, 0, 0);
            s->PutChar('}');
            break;

        case eFormatVectorOfSInt8:
            s->PutChar('{');
            offset = Dump (s, offset, eFormatDecimal, 1, item_byte_size, item_byte_size, LLDB_INVALID_ADDRESS, 0, 0);
            s->PutChar('}');
            break;

        case eFormatVectorOfUInt8:
            s->PutChar('{');
            offset = Dump (s, offset, eFormatHex, 1, item_byte_size, item_byte_size, LLDB_INVALID_ADDRESS, 0, 0);
            s->PutChar('}');
            break;

        case eFormatVectorOfSInt16:
            s->PutChar('{');
            offset = Dump (s, offset, eFormatDecimal, sizeof(uint16_t), item_byte_size / sizeof(uint16_t), item_byte_size / sizeof(uint16_t), LLDB_INVALID_ADDRESS, 0, 0);
            s->PutChar('}');
            break;

        case eFormatVectorOfUInt16:
            s->PutChar('{');
            offset = Dump (s, offset, eFormatHex,     sizeof(uint16_t), item_byte_size / sizeof(uint16_t), item_byte_size / sizeof(uint16_t), LLDB_INVALID_ADDRESS, 0, 0);
            s->PutChar('}');
            break;

        case eFormatVectorOfSInt32:
            s->PutChar('{');
            offset = Dump (s, offset, eFormatDecimal, sizeof(uint32_t), item_byte_size / sizeof(uint32_t), item_byte_size / sizeof(uint32_t), LLDB_INVALID_ADDRESS, 0, 0);
            s->PutChar('}');
            break;

        case eFormatVectorOfUInt32:
            s->PutChar('{');
            offset = Dump (s, offset, eFormatHex,     sizeof(uint32_t), item_byte_size / sizeof(uint32_t), item_byte_size / sizeof(uint32_t), LLDB_INVALID_ADDRESS, 0, 0);
            s->PutChar('}');
            break;

        case eFormatVectorOfSInt64:
            s->PutChar('{');
            offset = Dump (s, offset, eFormatDecimal, sizeof(uint64_t), item_byte_size / sizeof(uint64_t), item_byte_size / sizeof(uint64_t), LLDB_INVALID_ADDRESS, 0, 0);
            s->PutChar('}');
            break;

        case eFormatVectorOfUInt64:
            s->PutChar('{');
            offset = Dump (s, offset, eFormatHex,     sizeof(uint32_t), item_byte_size / sizeof(uint32_t), item_byte_size / sizeof(uint32_t), LLDB_INVALID_ADDRESS, 0, 0);
            s->PutChar('}');
            break;

        case eFormatVectorOfFloat32:
            s->PutChar('{');
            offset = Dump (s, offset, eFormatFloat,       4, item_byte_size / 4, item_byte_size / 4, LLDB_INVALID_ADDRESS, 0, 0);
            s->PutChar('}');
            break;

        case eFormatVectorOfFloat64:
            s->PutChar('{');
            offset = Dump (s, offset, eFormatFloat,       8, item_byte_size / 8, item_byte_size / 8, LLDB_INVALID_ADDRESS, 0, 0);
            s->PutChar('}');
            break;

        case eFormatVectorOfUInt128:
            s->PutChar('{');
            offset = Dump (s, offset, eFormatHex, 16, item_byte_size / 16, item_byte_size / 16, LLDB_INVALID_ADDRESS, 0, 0);
            s->PutChar('}');
            break;
        }
    }

    if (item_format == eFormatBytesWithASCII && offset > line_start_offset)
    {
        s->Printf("%*s", (num_per_line - (offset - line_start_offset)) * 3 + 2, "");
        Dump(s, line_start_offset, eFormatCharPrintable, 1, offset - line_start_offset, UINT32_MAX, LLDB_INVALID_ADDRESS, 0, 0);
    }
    return offset;  // Return the offset at which we ended up
}

//----------------------------------------------------------------------
// Dumps bytes from this object's data to the stream "s" starting
// "start_offset" bytes into this data, and ending with the byte
// before "end_offset". "base_addr" will be added to the offset
// into the dumped data when showing the offset into the data in the
// output information. "num_per_line" objects of type "type" will
// be dumped with the option to override the format for each object
// with "type_format". "type_format" is a printf style formatting
// string. If "type_format" is NULL, then an appropriate format
// string will be used for the supplied "type". If the stream "s"
// is NULL, then the output will be send to Log().
//----------------------------------------------------------------------
uint32_t
DataExtractor::PutToLog
(
    Log *log,
    uint32_t start_offset,
    uint32_t length,
    uint64_t base_addr,
    uint32_t num_per_line,
    DataExtractor::Type type,
    const char *format
) const
{
    if (log == NULL)
        return start_offset;

    uint32_t offset;
    uint32_t end_offset;
    uint32_t count;
    StreamString sstr;
    for (offset = start_offset, end_offset = offset + length, count = 0; ValidOffset(offset) && offset < end_offset; ++count)
    {
        if ((count % num_per_line) == 0)
        {
            // Print out any previous string
            if (sstr.GetSize() > 0)
            {
                log->Printf("%s", sstr.GetData());
                sstr.Clear();
            }
            // Reset string offset and fill the current line string with address:
            if (base_addr != LLDB_INVALID_ADDRESS)
                sstr.Printf("0x%8.8llx:", (uint64_t)(base_addr + (offset - start_offset)));
        }

        switch (type)
        {
            default:
            case TypeUInt8:   sstr.Printf (format ? format : " %2.2x", GetU8(&offset)); break;
            case TypeChar:
                {
                    char ch = GetU8(&offset);
                    sstr.Printf (format ? format : " %c",    isprint(ch) ? ch : ' ');
                }
                break;
            case TypeUInt16:  sstr.Printf (format ? format : " %4.4x",       GetU16(&offset)); break;
            case TypeUInt32:  sstr.Printf (format ? format : " %8.8x",       GetU32(&offset)); break;
            case TypeUInt64:  sstr.Printf (format ? format : " %16.16llx",   GetU64(&offset)); break;
            case TypePointer: sstr.Printf (format ? format : " 0x%llx",      GetAddress(&offset)); break;
            case TypeULEB128: sstr.Printf (format ? format : " 0x%llx",      GetULEB128(&offset)); break;
            case TypeSLEB128: sstr.Printf (format ? format : " %lld",        GetSLEB128(&offset)); break;
        }
    }

    if (sstr.GetSize() > 0)
        log->Printf("%s", sstr.GetData());

    return offset;  // Return the offset at which we ended up
}

//----------------------------------------------------------------------
// DumpUUID
//
// Dump out a UUID starting at 'offset' bytes into the buffer
//----------------------------------------------------------------------
void
DataExtractor::DumpUUID (Stream *s, uint32_t offset) const
{
    if (s)
    {
        const uint8_t *uuid_data = PeekData(offset, 16);
        if ( uuid_data )
        {
            lldb_private::UUID uuid(uuid_data, 16);
            uuid.Dump(s);
        }
        else
        {
            s->Printf("<not enough data for UUID at offset 0x%8.8x>", offset);
        }
    }
}

void
DataExtractor::DumpHexBytes (Stream *s, 
                             const void *src, 
                             size_t src_len, 
                             uint32_t bytes_per_line,
                             addr_t base_addr)
{
    DataExtractor data (src, src_len, eByteOrderLittle, 4);
    data.Dump (s, 
               0,               // Offset into "src"
               eFormatBytes,    // Dump as hex bytes
               1,               // Size of each item is 1 for single bytes
               src_len,         // Number of bytes
               bytes_per_line,  // Num bytes per line
               base_addr,       // Base address
               0, 0);           // Bitfield info
}

size_t
DataExtractor::Copy (DataExtractor &dest_data) const
{
    if (m_data_sp.get())
    {
        // we can pass along the SP to the data
        dest_data.SetData(m_data_sp);
    }
    else
    {
        const uint8_t *base_ptr = m_start;
        size_t data_size = GetByteSize();
        dest_data.SetData(DataBufferSP(new DataBufferHeap(base_ptr, data_size)));
    }
    return GetByteSize();
}

bool
DataExtractor::Append(DataExtractor& rhs)
{
    if (rhs.GetByteOrder() != GetByteOrder())
        return false;
    
    if (rhs.GetByteSize() == 0)
        return true;
    
    if (GetByteSize() == 0)
        return (rhs.Copy(*this) > 0);
    
    size_t bytes = GetByteSize() + rhs.GetByteSize();

    DataBufferHeap *buffer_heap_ptr = NULL;
    DataBufferSP buffer_sp(buffer_heap_ptr = new DataBufferHeap(bytes, 0));
    
    if (buffer_sp.get() == NULL || buffer_heap_ptr == NULL)
        return false;
    
    uint8_t* bytes_ptr = buffer_heap_ptr->GetBytes();
    
    memcpy(bytes_ptr, GetDataStart(), GetByteSize());
    memcpy(bytes_ptr + GetByteSize(), rhs.GetDataStart(), rhs.GetByteSize());
    
    SetData(buffer_sp);
    
    return true;
}

bool
DataExtractor::Append(void* buf, uint32_t length)
{
    if (buf == NULL)
        return false;
    
    if (length == 0)
        return true;
    
    size_t bytes = GetByteSize() + length;
    
    DataBufferHeap *buffer_heap_ptr = NULL;
    DataBufferSP buffer_sp(buffer_heap_ptr = new DataBufferHeap(bytes, 0));
    
    if (buffer_sp.get() == NULL || buffer_heap_ptr == NULL)
        return false;
    
    uint8_t* bytes_ptr = buffer_heap_ptr->GetBytes();
    
    if (GetByteSize() > 0)
        memcpy(bytes_ptr, GetDataStart(), GetByteSize());

    memcpy(bytes_ptr + GetByteSize(), buf, length);
    
    SetData(buffer_sp);
    
    return true;
}
