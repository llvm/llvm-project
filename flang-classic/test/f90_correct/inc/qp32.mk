#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp32  ########


qp32: run


build:  $(SRC)/qp32.f08
	-$(RM) qp32.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp32.f08 -o qp32.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp32.$(OBJX) check.$(OBJX) $(LIBS) -o qp32.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp32
	qp32.$(EXESUFFIX)

verify: ;
