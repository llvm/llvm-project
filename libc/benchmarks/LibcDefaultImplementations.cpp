#include "LibcFunctionPrototypes.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstddef>

namespace __llvm_libc {

extern void *memcpy(void *__restrict, const void *__restrict, size_t);
extern void *memset(void *, int, size_t);
extern void bzero(void *, size_t);
extern int memcmp(const void *, const void *, size_t);
extern int bcmp(const void *, const void *, size_t);

} // namespace __llvm_libc

// List of implementations to test.

using llvm::libc_benchmarks::BzeroConfiguration;
using llvm::libc_benchmarks::MemcmpConfiguration;
using llvm::libc_benchmarks::MemcpyConfiguration;
using llvm::libc_benchmarks::MemsetConfiguration;

llvm::ArrayRef<MemcpyConfiguration> getMemcpyConfigurations() {
  static constexpr MemcpyConfiguration kMemcpyConfigurations[] = {
      {__llvm_libc::memcpy, "__llvm_libc::memcpy"}};
  return llvm::makeArrayRef(kMemcpyConfigurations);
}
llvm::ArrayRef<MemcmpConfiguration> getMemcmpConfigurations() {
  static constexpr MemcmpConfiguration kMemcmpConfigurations[] = {
      {__llvm_libc::memcmp, "__llvm_libc::memcmp"}};
  return llvm::makeArrayRef(kMemcmpConfigurations);
}
llvm::ArrayRef<MemcmpConfiguration> getBcmpConfigurations() {
  static constexpr MemcmpConfiguration kBcmpConfigurations[] = {
      {__llvm_libc::bcmp, "__llvm_libc::bcmp"}};
  return llvm::makeArrayRef(kBcmpConfigurations);
}
llvm::ArrayRef<MemsetConfiguration> getMemsetConfigurations() {
  static constexpr MemsetConfiguration kMemsetConfigurations[] = {
      {__llvm_libc::memset, "__llvm_libc::memset"}};
  return llvm::makeArrayRef(kMemsetConfigurations);
}
llvm::ArrayRef<BzeroConfiguration> getBzeroConfigurations() {
  static constexpr BzeroConfiguration kBzeroConfigurations[] = {
      {__llvm_libc::bzero, "__llvm_libc::bzero"}};
  return llvm::makeArrayRef(kBzeroConfigurations);
}