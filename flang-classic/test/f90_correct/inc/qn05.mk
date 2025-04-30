#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn05  ########


qn05: run
	

build:  $(SRC)/qn05.f90
	-$(RM) qn05.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn05.f90 -o qn05.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn05.$(OBJX) check.$(OBJX) $(LIBS) -o qn05.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn05
	qn05.$(EXESUFFIX)

verify: ;

qn05.run: run

