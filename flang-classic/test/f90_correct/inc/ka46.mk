#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka46  ########


ka46: run
	

build:  $(SRC)/ka46.f
	-$(RM) ka46.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka46.f -o ka46.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka46.$(OBJX) check.$(OBJX) $(LIBS) -o ka46.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka46
	ka46.$(EXESUFFIX)

verify: ;

ka46.run: run

