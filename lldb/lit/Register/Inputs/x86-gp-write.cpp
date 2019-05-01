#include <cinttypes>
#include <cstdint>
#include <cstdio>

int main() {
  constexpr uint32_t fill = 0x0F0F0F0F;

  uint32_t eax, ebx, ecx, edx, esp, ebp, esi, edi;

  asm volatile(
    // save esp & ebp
    "movd    %%esp, %%mm0\n\t"
    "movd    %%ebp, %%mm1\n\t"
    "\n\t"
    "movl    %8, %%eax\n\t"
    "movl    %8, %%ebx\n\t"
    "movl    %8, %%ecx\n\t"
    "movl    %8, %%edx\n\t"
    "movl    %8, %%esp\n\t"
    "movl    %8, %%ebp\n\t"
    "movl    %8, %%esi\n\t"
    "movl    %8, %%edi\n\t"
    "\n\t"
    "int3\n\t"
    "\n\t"
    // first save new esp & ebp, and restore their original values, so that
    // we can output values via memory
    "movd    %%esp, %%mm2\n\t"
    "movd    %%ebp, %%mm3\n\t"
    "movd    %%mm0, %%esp\n\t"
    "movd    %%mm1, %%ebp\n\t"
    "\n\t"
    // output values via memory
    "movl    %%eax, %0\n\t"
    "movl    %%ebx, %1\n\t"
    "movl    %%ecx, %2\n\t"
    "movl    %%edx, %3\n\t"
    "movl    %%esi, %6\n\t"
    "movl    %%edi, %7\n\t"
    "\n\t"
    // output saved esp & ebp
    "movd    %%mm2, %4\n\t"
    "movd    %%mm3, %5\n\t"
    : "=m"(eax), "=m"(ebx), "=m"(ecx), "=m"(edx), "=a"(esp), "=b"(ebp),
      "=m"(esi), "=m"(edi)
    : "i"(fill)
    : "%ecx", "%edx", "%esp", "%ebp", "%esi", "%edi", "%mm0", "%mm1", "%mm2",
      "%mm3"
  );

  printf("eax = 0x%08" PRIx32 "\n", eax);
  printf("ebx = 0x%08" PRIx32 "\n", ebx);
  printf("ecx = 0x%08" PRIx32 "\n", ecx);
  printf("edx = 0x%08" PRIx32 "\n", edx);
  printf("esp = 0x%08" PRIx32 "\n", esp);
  printf("ebp = 0x%08" PRIx32 "\n", ebp);
  printf("esi = 0x%08" PRIx32 "\n", esi);
  printf("edi = 0x%08" PRIx32 "\n", edi);

  return 0;
}
