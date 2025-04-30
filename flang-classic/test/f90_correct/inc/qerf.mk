#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qerf  ########


qerf: run
	

build:  $(SRC)/qerf.f08
	-$(RM) qerf.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qerf.f08 -o qerf.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qerf.$(OBJX) check.$(OBJX) $(LIBS) -o qerf.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qerf 
	qerf.$(EXESUFFIX)

verify: ;


