add_custom_target(sys-socket-integration-tests)
add_dependencies(libc-integration-tests sys-socket-integration-tests)


add_integration_test(
  socket_send_recv_test
  SUITE
    sys-socket-integration-tests
  SRCS
    socket_send_recv_test.cpp
  DEPENDS
    libc.include.sys_socket
    libc.src.errno.errno
    libc.src.sys.socket.send
    libc.src.sys.socket.recv
    libc.src.sys.socket.socket
    libc.src.sys.socket.bind
    libc.src.sys.socket.connect
    libc.src.sys.socket.listen
    libc.src.sys.socket.accept
    libc.src.stdio.remove
    libc.src.unistd.close
    libc.src.unistd.fork
)
