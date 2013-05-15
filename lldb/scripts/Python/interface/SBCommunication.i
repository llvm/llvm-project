//===-- SWIG Interface for SBCommunication ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBCommunication
{
public:
    enum {
        eBroadcastBitDisconnected           = (1 << 0), ///< Sent when the communications connection is lost.
        eBroadcastBitReadThreadGotBytes     = (1 << 1), ///< Sent by the read thread when bytes become available.
        eBroadcastBitReadThreadDidExit      = (1 << 2), ///< Sent by the read thread when it exits to inform clients.
        eBroadcastBitReadThreadShouldExit   = (1 << 3), ///< Sent by clients that need to cancel the read thread.
        eBroadcastBitPacketAvailable        = (1 << 4), ///< Sent when data received makes a complete packet.
        eAllEventBits                       = 0xffffffff
    };

    typedef void (*ReadThreadBytesReceived) (void *baton, const void *src, size_t src_len);

    SBCommunication ();
    SBCommunication (const char * broadcaster_name);
   ~SBCommunication ();


    bool
    IsValid () const;

    lldb::SBBroadcaster
    GetBroadcaster ();

    static const char *GetBroadcasterClass();

    lldb::ConnectionStatus
    AdoptFileDesriptor (int fd, bool owns_fd);

    lldb::ConnectionStatus
    Connect (const char *url);

    lldb::ConnectionStatus
    Disconnect ();

    bool
    IsConnected () const;

    bool
    GetCloseOnEOF ();

    void
    SetCloseOnEOF (bool b);

    size_t
    Read (void *dst,
          size_t dst_len,
          uint32_t timeout_usec,
          lldb::ConnectionStatus &status);

    size_t
    Write (const void *src,
           size_t src_len,
           lldb::ConnectionStatus &status);

    bool
    ReadThreadStart ();

    bool
    ReadThreadStop ();

    bool
    ReadThreadIsRunning ();

    bool
    SetReadThreadBytesReceivedCallback (ReadThreadBytesReceived callback,
                                        void *callback_baton);
};

} // namespace lldb
