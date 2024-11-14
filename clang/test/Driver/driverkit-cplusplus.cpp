// RUN: %clang %s -target x86_64-apple-driverkit19.0 -fsyntax-only

#if __cplusplus != 201703L
#error DriverKit should be on C++17.
#endif

int main() { return 0; }
