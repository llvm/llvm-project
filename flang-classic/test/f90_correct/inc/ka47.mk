#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka47  ########


ka47: run
	

build:  $(SRC)/ka47.f
	-$(RM) ka47.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka47.f -o ka47.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka47.$(OBJX) check.$(OBJX) $(LIBS) -o ka47.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka47
	ka47.$(EXESUFFIX)

verify: ;

ka47.run: run

