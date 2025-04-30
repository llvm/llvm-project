#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn08a  ########


qn08a: run
	

build:  $(SRC)/qn08a.f90
	-$(RM) qn08a.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn08a.f90 -o qn08a.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn08a.$(OBJX) check.$(OBJX) $(LIBS) -o qn08a.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn08a
	qn08a.$(EXESUFFIX)

verify: ;

qn08a.run: run

