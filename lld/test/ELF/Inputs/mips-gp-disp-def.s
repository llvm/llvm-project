# We cannot create a shared library with defined _gp_disp symbol
# so we use a workaround - create a library with XXXXXXXX symbols
# and use 'sed' to replace it by _gp_disp right in the binary file.
  .data
  .globl XXXXXXXX
  .space 16
XXXXXXXX:
  .space 4
