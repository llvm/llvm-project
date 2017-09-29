// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -fasm-blocks -triple i386-apple-darwin10 -emit-llvm -o - | FileCheck %s

int gVar;
void t1() {
  // CHECK: add eax, dword ptr gVar[eax]
  __asm add eax, dword ptr gVar[eax]
  // CHECK: add dword ptr gVar[eax], eax
  __asm add dword ptr [eax+gVar], eax
  // CHECK: add ebx, dword ptr gVar[ebx + $$270]
  __asm add ebx, dword ptr gVar[271 - 82 + 81 + ebx]
  // CHECK: add dword ptr gVar[ebx + $$828], ebx
  __asm add dword ptr [ebx + gVar + 828], ebx
  // CHECK: add ecx, dword ptr gVar[ecx + ecx * $$4 + $$4590]
  __asm add ecx, dword ptr gVar[4590 + ecx + ecx*4]
  // CHECK: add dword ptr gVar[ecx + ecx * $$8 + $$73], ecx
  __asm add dword ptr [gVar + ecx + 45 + 23 - 53 + 60 - 2 + ecx*8], ecx
  // CHECK: add gVar[ecx + ebx + $$7], eax
  __asm add 1 + 1 + 2 + 3[gVar + ecx + ebx], eax
}

void t2() {
  int lVar;
  // CHECK: mov eax, dword ptr ${{[0-9]}}[eax]
  __asm mov eax, dword ptr lVar[eax]
  // CHECK: mov dword ptr ${{[0-9]}}[eax], eax
  __asm mov dword ptr [eax+lVar], eax
  // CHECK: mov ebx, dword ptr ${{[0-9]}}[ebx + $$270]
  __asm mov ebx, dword ptr lVar[271 - 82 + 81 + ebx]
  // CHECK: mov dword ptr ${{[0-9]}}[ebx + $$828], ebx
  __asm mov dword ptr [ebx + lVar + 828], ebx
  // CHECK: mov ${{[0-9]}}[ebx + $$47], eax
  __asm mov 5 + 8 + 13 + 21[lVar + ebx], eax
}

