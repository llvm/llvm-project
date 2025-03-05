#include "LibcFunctionPrototypes.h"
#include "src/__support/macros/config.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstddef>

namespace LIBC_NAMESPACE_DECL {

extern void *memcpy(void *__restrict, const void *__restrict, size_t);
extern void *memmove(void *, const void *, size_t);
extern void *memset(void *, int, size_t);
extern void bzero(void *, size_t);
extern int memcmp(const void *, const void *, size_t);
extern int bcmp(const void *, const void *, size_t);

} // namespace LIBC_NAMESPACE_DECL

// List of implementations to test.

using llvm::libc_benchmarks::BzeroConfiguration;
using llvm::libc_benchmarks::MemcmpOrBcmpConfiguration;
using llvm::libc_benchmarks::MemcpyConfiguration;
using llvm::libc_benchmarks::MemmoveConfiguration;
using llvm::libc_benchmarks::MemsetConfiguration;

llvm::ArrayRef<MemcpyConfiguration> getMemcpyConfigurations() {
  static constexpr MemcpyConfiguration kMemcpyConfigurations[] = {
      {LIBC_NAMESPACE::memcpy, "LIBC_NAMESPACE::memcpy"}};
  return llvm::ArrayRef(kMemcpyConfigurations);
}
llvm::ArrayRef<MemmoveConfiguration> getMemmoveConfigurations() {
  static constexpr MemmoveConfiguration kMemmoveConfigurations[] = {
      {LIBC_NAMESPACE::memmove, "LIBC_NAMESPACE::memmove"}};
  return llvm::ArrayRef(kMemmoveConfigurations);
}
llvm::ArrayRef<MemcmpOrBcmpConfiguration> getMemcmpConfigurations() {
  static constexpr MemcmpOrBcmpConfiguration kMemcmpConfiguration[] = {
      {LIBC_NAMESPACE::memcmp, "LIBC_NAMESPACE::memcmp"}};
  return llvm::ArrayRef(kMemcmpConfiguration);
}
llvm::ArrayRef<MemcmpOrBcmpConfiguration> getBcmpConfigurations() {
  static constexpr MemcmpOrBcmpConfiguration kBcmpConfigurations[] = {
      {LIBC_NAMESPACE::bcmp, "LIBC_NAMESPACE::bcmp"}};
  return llvm::ArrayRef(kBcmpConfigurations);
}
llvm::ArrayRef<MemsetConfiguration> getMemsetConfigurations() {
  static constexpr MemsetConfiguration kMemsetConfigurations[] = {
      {LIBC_NAMESPACE::memset, "LIBC_NAMESPACE::memset"}};
  return llvm::ArrayRef(kMemsetConfigurations);
}
llvm::ArrayRef<BzeroConfiguration> getBzeroConfigurations() {
  static constexpr BzeroConfiguration kBzeroConfigurations[] = {
      {LIBC_NAMESPACE::bzero, "LIBC_NAMESPACE::bzero"}};
  return llvm::ArrayRef(kBzeroConfigurations);
}
