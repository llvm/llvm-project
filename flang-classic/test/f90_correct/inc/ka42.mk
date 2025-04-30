#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka42  ########


ka42: run
	

build:  $(SRC)/ka42.f
	-$(RM) ka42.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka42.f -o ka42.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka42.$(OBJX) check.$(OBJX) $(LIBS) -o ka42.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka42
	ka42.$(EXESUFFIX)

verify: ;

ka42.run: run

