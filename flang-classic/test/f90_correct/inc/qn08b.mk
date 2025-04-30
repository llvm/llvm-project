#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn08b  ########


qn08b: run
	

build:  $(SRC)/qn08b.f90
	-$(RM) qn08b.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn08b.f90 -o qn08b.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn08b.$(OBJX) check.$(OBJX) $(LIBS) -o qn08b.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn08b
	qn08b.$(EXESUFFIX)

verify: ;

qn08b.run: run

