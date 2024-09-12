#ifndef CLANG_CIR_TYPEEVALUATIONKIND_H
#define CLANG_CIR_TYPEEVALUATIONKIND_H

namespace cir {

// FIXME: for now we are reusing this from lib/Clang/CIRGenFunction.h, which
// isn't available in the include dir. Same for getEvaluationKind below.
enum TypeEvaluationKind { TEK_Scalar, TEK_Complex, TEK_Aggregate };

} // namespace cir

#endif // CLANG_CIR_TYPEEVALUATIONKIND_H
