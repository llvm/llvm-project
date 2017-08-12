// REQUIRES: mips-registered-target
// RUN: %clang --target=mips64-unknown-linux -S -mmadd4    %s -o -| FileCheck %s -check-prefix=MADD4
// RUN: %clang --target=mips64-unknown-linux -S -mno-madd4 %s -o -| FileCheck %s -check-prefix=NOMADD4
// RUN: %clang --target=mips64-unknown-linux -S -mmadd4    -fno-honor-nans %s -o -| FileCheck %s -check-prefix=MADD4-NONAN
// RUN: %clang --target=mips64-unknown-linux -S -mno-madd4 -fno-honor-nans %s -o -| FileCheck %s -check-prefix=NOMADD4-NONAN
 
float madd_s (float f, float g, float h)
{
  return (f * g) + h;
}
// MADD4:   madd.s
// NOMADD4: mul.s
// NOMADD4: add.s

float msub_s (float f, float g, float h)
{
  return (f * g) - h;
}
// MADD4:   msub.s
// NOMADD4: mul.s
// NOMADD4: sub.s

double madd_d (double f, double g, double h)
{
  return (f * g) + h;
}
// MADD4:   madd.d
// NOMADD4: mul.d
// NOMADD4: add.d

double msub_d (double f, double g, double h)
{
  return (f * g) - h;
}
// MADD4:   msub.d
// NOMADD4: mul.d
// NOMADD4: sub.d


float nmadd_s (float f, float g, float h)
{
  // FIXME: Zero has been explicitly placed to force generation of a positive
  // zero in IR until pattern used to match this instruction is changed to
  // comply with negative zero as well.
  return 0-((f * g) + h);
}
// MADD4-NONAN:   nmadd.s
// NOMADD4-NONAN: mul.s
// NOMADD4-NONAN: add.s
// NOMADD4-NONAN: sub.s

float nmsub_s (float f, float g, float h)
{
  // FIXME: Zero has been explicitly placed to force generation of a positive
  // zero in IR until pattern used to match this instruction is changed to
  // comply with negative zero as well.
  return 0-((f * g) - h);
}
// MADD4-NONAN:   nmsub.s
// NOMADD4-NONAN: mul.s
// NOMADD4-NONAN: sub.s
// NOMADD4-NONAN: sub.s

double nmadd_d (double f, double g, double h)
{
  // FIXME: Zero has been explicitly placed to force generation of a positive
  // zero in IR until pattern used to match this instruction is changed to
  // comply with negative zero as well.
  return 0-((f * g) + h);
}
// MADD4-NONAN:   nmadd.d
// NOMADD4-NONAN: mul.d
// NOMADD4-NONAN: add.d
// NOMADD4-NONAN: sub.d

double nmsub_d (double f, double g, double h)
{
  // FIXME: Zero has been explicitly placed to force generation of a positive
  // zero in IR until pattern used to match this instruction is changed to
  // comply with negative zero as well.
  return 0-((f * g) - h);
}
// MADD4-NONAN:   nmsub.d
// NOMADD4-NONAN: mul.d
// NOMADD4-NONAN: sub.d
// NOMADD4-NONAN: sub.d

