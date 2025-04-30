#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka43  ########


ka43: run
	

build:  $(SRC)/ka43.f
	-$(RM) ka43.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka43.f -o ka43.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka43.$(OBJX) check.$(OBJX) $(LIBS) -o ka43.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka43
	ka43.$(EXESUFFIX)

verify: ;

ka43.run: run

