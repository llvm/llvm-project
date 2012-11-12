//===-- Connection.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Connection_h_
#define liblldb_Connection_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Connection Connection.h "lldb/Core/Connection.h"
/// @brief A communication connection class.
///
/// A class that implements that actual communication functions for
/// connecting/disconnecting, reading/writing, and waiting for bytes
/// to become available from a two way communication connection.
///
/// This class is designed to only do very simple communication
/// functions. Instances can be instantiated and given to a
/// Communication class to perform communications where clients can
/// listen for broadcasts, and perform other higher level communications.
//----------------------------------------------------------------------
class Connection
{
public:
    //------------------------------------------------------------------
    /// Default constructor
    //------------------------------------------------------------------
    Connection ();

    //------------------------------------------------------------------
    /// Virtual destructor since this class gets subclassed and handed
    /// to a Communication object.
    //------------------------------------------------------------------
    virtual
    ~Connection ();

    //------------------------------------------------------------------
    /// Connect using the connect string \a url.
    ///
    /// @param[in] url
    ///     A string that contains all information needed by the
    ///     subclass to connect to another client.
    ///
    /// @param[out] error_ptr
    ///     A pointer to an error object that should be given an
    ///     approriate error value if this method returns false. This
    ///     value can be NULL if the error value should be ignored.
    ///
    /// @return
    ///     \b True if the connect succeeded, \b false otherwise. The
    ///     internal error object should be filled in with an
    ///     appropriate value based on the result of this function.
    ///
    /// @see Error& Communication::GetError ();
    //------------------------------------------------------------------
    virtual lldb::ConnectionStatus
    Connect (const char *url, Error *error_ptr) = 0;

    //------------------------------------------------------------------
    /// Disconnect the communications connection if one is currently
    /// connected.
    ///
    /// @param[out] error_ptr
    ///     A pointer to an error object that should be given an
    ///     approriate error value if this method returns false. This
    ///     value can be NULL if the error value should be ignored.
    ///
    /// @return
    ///     \b True if the disconnect succeeded, \b false otherwise. The
    ///     internal error object should be filled in with an
    ///     appropriate value based on the result of this function.
    ///
    /// @see Error& Communication::GetError ();
    //------------------------------------------------------------------
    virtual lldb::ConnectionStatus
    Disconnect (Error *error_ptr) = 0;

    //------------------------------------------------------------------
    /// Check if the connection is valid.
    ///
    /// @return
    ///     \b True if this object is currently connected, \b false
    ///     otherwise.
    //------------------------------------------------------------------
    virtual bool
    IsConnected () const = 0;

    //------------------------------------------------------------------
    /// The read function that attempts to read from the connection.
    ///
    /// @param[in] dst
    ///     A destination buffer that must be at least \a dst_len bytes
    ///     long.
    ///
    /// @param[in] dst_len
    ///     The number of bytes to attempt to read, and also the max
    ///     number of bytes that can be placed into \a dst.
    ///
    /// @param[out] error_ptr
    ///     A pointer to an error object that should be given an
    ///     approriate error value if this method returns zero. This
    ///     value can be NULL if the error value should be ignored.
    ///
    /// @return
    ///     The number of bytes actually read.
    ///
    /// @see size_t Communication::Read (void *, size_t, uint32_t);
    //------------------------------------------------------------------
    virtual size_t
    Read (void *dst, 
          size_t dst_len, 
          uint32_t timeout_usec,
          lldb::ConnectionStatus &status, 
          Error *error_ptr) = 0;

    //------------------------------------------------------------------
    /// The actual write function that attempts to write to the
    /// communications protocol.
    ///
    /// Subclasses must override this function.
    ///
    /// @param[in] src
    ///     A source buffer that must be at least \a src_len bytes
    ///     long.
    ///
    /// @param[in] src_len
    ///     The number of bytes to attempt to write, and also the
    ///     number of bytes are currently available in \a src.
    ///
    /// @param[out] error_ptr
    ///     A pointer to an error object that should be given an
    ///     approriate error value if this method returns zero. This
    ///     value can be NULL if the error value should be ignored.
    ///
    /// @return
    ///     The number of bytes actually Written.
    //------------------------------------------------------------------
    virtual size_t
    Write (const void *buffer, size_t length, lldb::ConnectionStatus &status, Error *error_ptr) = 0;

private:
    //------------------------------------------------------------------
    // For Connection only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (Connection);
};

} // namespace lldb_private

#endif  // liblldb_Connection_h_
