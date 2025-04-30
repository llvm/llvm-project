#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn08  ########


qcall: run
	

build:  $(SRC)/qcall.f08
	-$(RM) qcall.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qcall.f08 -o qcall.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qcall.$(OBJX) check.$(OBJX) $(LIBS) -o qcall.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qcall
	qcall.$(EXESUFFIX)

verify: ;

qcall.run: run

