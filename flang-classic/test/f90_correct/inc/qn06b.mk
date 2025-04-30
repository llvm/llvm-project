#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn06b  ########


qn06b: run
	

build:  $(SRC)/qn06b.f90
	-$(RM) qn06b.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn06b.f90 -o qn06b.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn06b.$(OBJX) check.$(OBJX) $(LIBS) -o qn06b.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn06b
	qn06b.$(EXESUFFIX)

verify: ;

qn06b.run: run

