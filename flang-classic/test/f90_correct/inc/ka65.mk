#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka65  ########


ka65: run
	

build:  $(SRC)/ka65.f90
	-$(RM) ka65.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka65.f90 -o ka65.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka65.$(OBJX) check.$(OBJX) $(LIBS) -o ka65.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka65
	ka65.$(EXESUFFIX)

verify: ;

ka65.run: run

