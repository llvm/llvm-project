#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qmaxloc  ########


qmaxloc: run
	

build:  $(SRC)/qmaxloc.f08
	-$(RM) qmaxloc.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qmaxloc.f08 -o qmaxloc.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qmaxloc.$(OBJX) check.$(OBJX) $(LIBS) -o qmaxloc.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qmaxloc 
	qmaxloc.$(EXESUFFIX)

verify: ;


