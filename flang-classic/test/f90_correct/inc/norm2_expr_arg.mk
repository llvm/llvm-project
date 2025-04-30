#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
EXE=norm2_expr_arg.$(EXESUFFIX)

build:  $(SRC)/norm2_expr_arg.F90
	-$(RM) norm2_expr_arg.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	-$(RM) $(OBJ)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	@echo ------------------------------------ building test $@
	$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/norm2_expr_arg.F90 check.$(OBJX) -o norm2_expr_arg.$(EXESUFFIX)
	$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/norm2_expr_arg.F90 check.$(OBJX) -r8 -o norm2_expr_argR8.$(EXESUFFIX)

run:
	@echo ------------------------------------ executing test norm2_expr_arg
	norm2_expr_arg.$(EXESUFFIX)
	norm2_expr_argR8.$(EXESUFFIX)

verify: ;

norm2_expr_arg.run: run
