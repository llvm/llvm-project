#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka69  ########


ka69: run
	

build:  $(SRC)/ka69.f90
	-$(RM) ka69.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka69.f90 -o ka69.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka69.$(OBJX) check.$(OBJX) $(LIBS) -o ka69.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka69
	ka69.$(EXESUFFIX)

verify: ;

ka69.run: run

