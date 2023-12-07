// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

typedef unsigned long word_t;
typedef unsigned long u64_t;
typedef unsigned int u32_t;

class ioapic_redir_t {
public:
 union {
  struct {
   word_t vector : 8;

   word_t delivery_mode : 3;
   word_t dest_mode : 1;

   word_t delivery_status : 1;
   word_t polarity : 1;
   word_t irr : 1;
   word_t trigger_mode : 1;

   word_t mask : 1;
   word_t _pad0 : 15;

   word_t dest : 8;
  };
  volatile u32_t raw[2];
  volatile u64_t raw64;
 };
};

struct ioapic_shadow_struct
{
 ioapic_redir_t redirs[24];
} ioapic_shadow[16];

void init_ioapic(unsigned long ioapic_id)
{
     ioapic_redir_t entry;
     ioapic_shadow[ioapic_id].redirs[3] = entry;
}

// CHECK: call void @llvm.memcpy
