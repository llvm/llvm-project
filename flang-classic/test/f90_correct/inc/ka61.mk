#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka61  ########


ka61: run
	

build:  $(SRC)/ka61.f
	-$(RM) ka61.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka61.f -o ka61.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka61.$(OBJX) check.$(OBJX) $(LIBS) -o ka61.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka61
	ka61.$(EXESUFFIX)

verify: ;

ka61.run: run

