// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:DisplayLoadedSummaries=true \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -triple i686-unknown-linux 2>&1 | FileCheck %s

// We test here that functions from socket.h are added when sockaddr is not a
// transparent union of other sockaddr_ pointers. This is the case in C++.

// CHECK: Loaded summary for: int accept(int socket, struct sockaddr *address, socklen_t *address_len)
// CHECK: Loaded summary for: int bind(int socket, const struct sockaddr *address, socklen_t address_len)
// CHECK: Loaded summary for: int getpeername(int socket, struct sockaddr *address, socklen_t *address_len)
// CHECK: Loaded summary for: int getsockname(int socket, struct sockaddr *address, socklen_t *address_len)
// CHECK: Loaded summary for: int connect(int socket, const struct sockaddr *address, socklen_t address_len)
// CHECK: Loaded summary for: ssize_t recvfrom(int socket, void *buffer, size_t length, int flags, struct sockaddr *address, socklen_t *address_len)
// CHECK: Loaded summary for: ssize_t sendto(int socket, const void *message, size_t length, int flags, const struct sockaddr *dest_addr, socklen_t dest_len)

struct sockaddr;
using socklen_t = unsigned;
int accept(int socket, struct sockaddr *address, socklen_t *address_len);
int bind(int socket, const struct sockaddr *address, socklen_t address_len);
int getpeername(int socket, struct sockaddr *address, socklen_t *address_len);
int getsockname(int socket, struct sockaddr *address, socklen_t *address_len);
int connect(int socket, const struct sockaddr *address, socklen_t address_len);
typedef decltype(sizeof(int)) size_t;
typedef size_t ssize_t;
ssize_t recvfrom(int socket, void *buffer, size_t length, int flags, struct sockaddr *address, socklen_t *address_len);
ssize_t sendto(int socket, const void *message, size_t length, int flags, const struct sockaddr *dest_addr, socklen_t dest_len);

// Must have at least one call expression to initialize the summary map.
int bar(void);
void foo() {
  bar();
}
