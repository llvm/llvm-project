#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka60  ########


ka60: run
	

build:  $(SRC)/ka60.f
	-$(RM) ka60.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka60.f -o ka60.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka60.$(OBJX) check.$(OBJX) $(LIBS) -o ka60.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka60
	ka60.$(EXESUFFIX)

verify: ;

ka60.run: run

