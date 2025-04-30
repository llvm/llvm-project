#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn02b  ########


qn02b: run
	

build:  $(SRC)/qn02b.f90
	-$(RM) qn02b.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn02b.f90 -o qn02b.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn02b.$(OBJX) check.$(OBJX) $(LIBS) -o qn02b.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn02b
	qn02b.$(EXESUFFIX)

verify: ;

qn02b.run: run

