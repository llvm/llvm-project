#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn02d  ########


qn02d: run
	

build:  $(SRC)/qn02d.f90
	-$(RM) qn02d.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn02d.f90 -o qn02d.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn02d.$(OBJX) check.$(OBJX) $(LIBS) -o qn02d.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn02d
	qn02d.$(EXESUFFIX)

verify: ;

qn02d.run: run

