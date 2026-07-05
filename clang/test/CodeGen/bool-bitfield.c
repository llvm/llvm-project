// RUN: %clang_cc1 -emit-llvm %s -o %t

// From GCC PR19331
struct SysParams
{
 unsigned short tag;
 unsigned short version;
 unsigned int seqnum;
 int contrast;
 int igain_1, igain_2;
 int oattn_1, oattn_2;
 int max_out_vltg_1, max_out_vltg_2;
 int max_mains_current;
 int meters_mode;
 int input_select;
 _Bool input_parallelch2:1;
 _Bool cliplmt_ch1:1;
 _Bool cliplmt_ch2:1;
 _Bool gate_ch1:1;
 _Bool gate_ch2:1;
 _Bool mute_ch1:1;
 _Bool mute_ch2:1;
 _Bool brownout:1;
 _Bool power_on:1;
 _Bool pwrup_mute:1;
 _Bool keylock:1;
 _Bool dsp_ch1:1;
 _Bool dsp_ch2:1;
 int dsp_preset;
 long unlock_code;
};
extern struct SysParams params;

void foo(void *);
void kcmd_setParams(void)
{
 struct {
  unsigned char igain_1;
  unsigned char igain_2;
  unsigned char max_out_vltg_1;
  unsigned char max_out_vltg_2;
  unsigned char max_imains;
  unsigned char cliplmt_ch1:1;
  unsigned char cliplmt_ch2:1;
  unsigned char gate_ch1:1;
  unsigned char gate_ch2:1;
 } msg;
 foo(&msg);
 params.cliplmt_ch1 = msg.cliplmt_ch1;
 params.cliplmt_ch2 = msg.cliplmt_ch2;
 params.gate_ch1 = msg.gate_ch1;
 params.gate_ch2 = msg.gate_ch2;
}

