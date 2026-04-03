//===-- Unittests for shared RPC server -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "shared/rpc.h"
#include "shared/rpc_server.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

TEST(LlvmLibcSharedRpcTest, TestIncrement) {
  constexpr uint32_t port_count = 1;
  constexpr uint32_t lane_size = 1;
  alignas(64)
      uint8_t buffer[::rpc::Server::allocation_size(lane_size, port_count)];

  ::rpc::Server server(port_count, buffer);
  ::rpc::Client client(port_count, buffer);

  // Client side: open a port and send data
  auto client_port = client.open<LIBC_TEST_INCREMENT>();
  uint64_t data = 42;
  client_port.send(
      [&](::rpc::Buffer *buffer, uint32_t) { buffer->data[0] = data; });

  // Server side: handle the port
  auto server_port = server.try_open(lane_size);
  ASSERT_TRUE(server_port.has_value());
  ::rpc::RPCStatus status =
      shared::handle_libc_opcodes(*server_port, lane_size);
  EXPECT_EQ(status, ::rpc::RPC_SUCCESS);

  // Client side: receive the response
  client_port.recv(
      [&](::rpc::Buffer *buffer, uint32_t) { data = buffer->data[0]; });

  EXPECT_EQ(data, 43UL);
}

} // namespace LIBC_NAMESPACE_DECL
