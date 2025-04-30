#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn08  ########


qn08: run
	

build:  $(SRC)/qn08.f90
	-$(RM) qn08.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn08.f90 -o qn08.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn08.$(OBJX) check.$(OBJX) $(LIBS) -o qn08.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn08
	qn08.$(EXESUFFIX)

verify: ;

qn08.run: run

