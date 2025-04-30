#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qfloor  ########


qfloor: run
	

build:  $(SRC)/qfloor.f08
	-$(RM) qfloor.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qfloor.f08 -o qfloor.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qfloor.$(OBJX) check.$(OBJX) $(LIBS) -o qfloor.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qfloor 
	qfloor.$(EXESUFFIX)

verify: ;


