#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn08d  ########


qn08d: run
	

build:  $(SRC)/qn08d.f90
	-$(RM) qn08d.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn08d.f90 -o qn08d.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn08d.$(OBJX) check.$(OBJX) $(LIBS) -o qn08d.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn08d
	qn08d.$(EXESUFFIX)

verify: ;

qn08d.run: run

