//===-- DataBufferMemoryMap.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DataBufferMemoryMap_h_
#define liblldb_DataBufferMemoryMap_h_
#if defined(__cplusplus)


#include "lldb/lldb-private.h"
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/Error.h"
#include <string>

namespace lldb_private {

//----------------------------------------------------------------------
/// @class DataBufferMemoryMap DataBufferMemoryMap.h "lldb/Core/DataBufferMemoryMap.h"
/// @brief A subclass of DataBuffer that memory maps data.
///
/// This class memory maps data and stores any needed data for the
/// memory mapping in its internal state. Memory map requests are not
/// required to have any alignment or size constraints, this class will
/// work around any host OS issues regarding such things.
///
/// This class is designed to allow pages to be faulted in as needed and
/// works well data from large files that won't be accessed all at once.
//----------------------------------------------------------------------
class DataBufferMemoryMap : public DataBuffer
{
public:
    //------------------------------------------------------------------
    /// Default Constructor
    //------------------------------------------------------------------
    DataBufferMemoryMap ();

    //------------------------------------------------------------------
    /// Destructor.
    ///
    /// Virtual destructor since this class inherits from a pure virtual
    /// base class #DataBuffer.
    //------------------------------------------------------------------
    virtual
    ~DataBufferMemoryMap ();

    //------------------------------------------------------------------
    /// Reverts this object to an empty state by unmapping any memory
    /// that is currently owned.
    //------------------------------------------------------------------
    void
    Clear ();

    //------------------------------------------------------------------
    /// @copydoc DataBuffer::GetBytes()
    //------------------------------------------------------------------
    virtual uint8_t *
    GetBytes ();

    //------------------------------------------------------------------
    /// @copydoc DataBuffer::GetBytes() const
    //------------------------------------------------------------------
    virtual const uint8_t *
    GetBytes () const;

    //------------------------------------------------------------------
    /// @copydoc DataBuffer::GetByteSize() const
    //------------------------------------------------------------------
    virtual lldb::offset_t
    GetByteSize () const;

    //------------------------------------------------------------------
    /// Error get accessor.
    ///
    /// @return
    ///     A const reference to Error object in case memory mapping
    ///     fails.
    //------------------------------------------------------------------
    const Error &
    GetError() const;

    //------------------------------------------------------------------
    /// Memory map all or part of a file.
    ///
    /// Memory map \a length bytes from \a file starting \a offset
    /// bytes into the file. If \a length is set to \c SIZE_MAX,
    /// then map as many bytes as possible.
    ///
    /// @param[in] file
    ///     The file specification from which to map data.
    ///
    /// @param[in] offset
    ///     The offset in bytes from the beginning of the file where
    ///     memory mapping should begin.
    ///
    /// @param[in] length
    ///     The size in bytes that should be mapped starting \a offset
    ///     bytes into the file. If \a length is \c SIZE_MAX, map
    ///     as many bytes as possible.
    ///
    /// @return
    ///     The number of bytes mapped starting from the \a offset.
    //------------------------------------------------------------------
    size_t
    MemoryMapFromFileSpec (const FileSpec* file,
                           lldb::offset_t offset = 0,
                           lldb::offset_t length = SIZE_MAX,
                           bool writeable = false);

    //------------------------------------------------------------------
    /// Memory map all or part of a file.
    ///
    /// Memory map \a length bytes from an opened file descriptor \a fd
    /// starting \a offset bytes into the file. If \a length is set to
    /// \c SIZE_MAX, then map as many bytes as possible.
    ///
    /// @param[in] fd
    ///     The posix file descriptor for an already opened file
    ///     from which to map data.
    ///
    /// @param[in] offset
    ///     The offset in bytes from the beginning of the file where
    ///     memory mapping should begin.
    ///
    /// @param[in] length
    ///     The size in bytes that should be mapped starting \a offset
    ///     bytes into the file. If \a length is \c SIZE_MAX, map
    ///     as many bytes as possible.
    ///
    /// @return
    ///     The number of bytes mapped starting from the \a offset.
    //------------------------------------------------------------------
    size_t
    MemoryMapFromFileDescriptor (int fd, 
                                 lldb::offset_t offset,
                                 lldb::offset_t length,
                                 bool write,
                                 bool fd_is_file);

protected:
    //------------------------------------------------------------------
    // Classes that inherit from DataBufferMemoryMap can see and modify these
    //------------------------------------------------------------------
    uint8_t * m_mmap_addr;  ///< The actual pointer that was returned from \c mmap()
    size_t m_mmap_size;     ///< The actual number of bytes that were mapped when \c mmap() was called
    uint8_t *m_data;        ///< The data the user requested somewhere within the memory mapped data.
    lldb::offset_t m_size;  ///< The size of the data the user got when data was requested

private:
    DISALLOW_COPY_AND_ASSIGN (DataBufferMemoryMap);
};

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // liblldb_DataBufferMemoryMap_h_
