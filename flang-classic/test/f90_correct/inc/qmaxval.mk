#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qmaxval  ########


qmaxval: run
	

build:  $(SRC)/qmaxval.f08
	-$(RM) qmaxval.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qmaxval.f08 -o qmaxval.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qmaxval.$(OBJX) check.$(OBJX) $(LIBS) -o qmaxval.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qmaxval 
	qmaxval.$(EXESUFFIX)

verify: ;


