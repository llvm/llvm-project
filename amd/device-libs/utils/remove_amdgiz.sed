# Input: amdgcn--amdhsa-amdgiz
# Output: amdgcn--amdhsa

#######################
# change target triple
#######################

# amdgcn--amdhsa-amdgiz -> amdgcn--amdhsa
/target triple/s/\"amdgcn--amdhsa-amdgiz\"/\"amdgcn--amdhsa\"/

#####################
# change data layout
#####################

# p:64:64 -> p:32:32
/target datalayout/s/-p:64:64-/-p:32:32-/
/target datalayout/s/-p4:32:32-/-p4:64:64-/
/target datalayout/s/-n32:64-A5/-n32:64/

#############################
# change intrinsic functions
#############################

# change p0i32 -> p4i32
s/\.p0i32(/\.p4i32(/g

###########################
# change pointer addrspace
###########################

# change addrspace(0) -> addrspace(4)
s/addrspace(0)\*/addrspace(4)*/g

# change addrspace(5) -> addrspace(0)
s/addrspace(5)\*/addrspace(0)*/g

