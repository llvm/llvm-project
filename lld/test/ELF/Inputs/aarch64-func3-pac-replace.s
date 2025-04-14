// Declare file properties exclusively with aarch64 build attributes.

.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_PAC, 1

.text
.globl func3
.type func3,@function
func3:
  ret
