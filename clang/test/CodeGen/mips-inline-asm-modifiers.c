// RUN: %clang_cc1 -triple mipsel-unknown-linux -o - -emit-llvm %s \
// RUN: | FileCheck %s

// This checks that the frontend will accept inline asm operand modifiers

int printf(const char*, ...);

typedef int v4i32 __attribute__((vector_size(16)));

  // CHECK: %{{[0-9]+}} = call i32 asm ".set noreorder;\0Alw    $0,$1;\0A.set reorder;\0A", "=r,*m,~{$1}"(ptr elementtype(i32) getelementptr inbounds (i32, ptr @b, i32 {{[0-9]+}})) #2,
  // CHECK: %{{[0-9]+}} = call i32 asm "lw    $0,${1:D};\0A", "=r,*m,~{$1}"(ptr elementtype(i32) getelementptr inbounds (i32, ptr @b, i32 {{[0-9]+}})) #2,
  // CHECK: %{{[0-9]+}} = call <4 x i32> asm "ldi.w ${0:w},1", "=f,~{$1}"
int b[8] = {0,1,2,3,4,5,6,7};
int  main(void)
{
  int i;
  v4i32 v4i32_r;

  // The first word. Notice, no 'D'
  {asm (
  ".set noreorder;\n"
  "lw    %0,%1;\n"
  ".set reorder;\n"
  : "=r" (i)
  : "m" (*(b+4)));}

  printf("%d\n",i);

  // The second word
  {asm (
  "lw    %0,%D1;\n"
  : "=r" (i)
  : "m" (*(b+4))
  );}

  // MSA registers
  {asm ("ldi.w %w0,1" : "=f" (v4i32_r));}

  printf("%d\n",i);

  return 1;
}
