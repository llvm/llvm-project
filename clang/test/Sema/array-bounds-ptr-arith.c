// RUN: %clang_cc1 -verify -Warray-bounds-pointer-arithmetic -fstrict-flex-arrays=1 %s

// Test case from PR10615
struct ext2_super_block{
  unsigned char s_uuid[8]; // expected-note {{declared here}}
};
void* ext2_statfs (struct ext2_super_block *es,int a)
{
	 return (void *)es->s_uuid + sizeof(int); // no-warning
}
void* broken (struct ext2_super_block *es,int a)
{
	 return (void *)es->s_uuid + 80; // expected-warning {{refers past the end of the array}}
}
