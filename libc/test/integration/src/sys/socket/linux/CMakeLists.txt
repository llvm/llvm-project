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
    libc.src.sys.socket.socketpair
    libc.src.stdio.remove
    libc.src.stdio.puts
    libc.src.unistd.close
    libc.src.unistd.fork
)

add_integration_test(
  socket_sendto_recvfrom_test
  SUITE
    sys-socket-integration-tests
  SRCS
    socket_sendto_recvfrom_test.cpp
  DEPENDS
    libc.include.sys_socket
    libc.src.errno.errno
    libc.src.sys.socket.sendto
    libc.src.sys.socket.recvfrom
    libc.src.sys.socket.socketpair
    libc.src.stdio.remove
    libc.src.stdio.puts
    libc.src.unistd.close
    libc.src.unistd.fork
)

add_integration_test(
  socket_sendmsg_recvmsg_test
  SUITE
    sys-socket-integration-tests
  SRCS
    socket_sendmsg_recvmsg_test.cpp
  DEPENDS
    libc.include.sys_socket
    libc.src.errno.errno
    libc.src.sys.socket.sendmsg
    libc.src.sys.socket.recvmsg
    libc.src.sys.socket.socketpair
    libc.src.stdio.remove
    libc.src.stdio.puts
    libc.src.unistd.close
    libc.src.unistd.fork
)
