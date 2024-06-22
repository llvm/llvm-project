
#ifndef CIR_AAARCH64_H
#define CIR_AAARCH64_H

namespace cir {

/// The ABI kind for AArch64 targets.
enum class AArch64ABIKind {
  AAPCS = 0,
  DarwinPCS,
  Win64,
  AAPCSSoft,
};

} // namespace cir

#endif // CIR_AAARCH64_H
