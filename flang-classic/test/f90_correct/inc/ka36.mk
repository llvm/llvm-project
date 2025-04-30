#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka36  ########


ka36: run
	

build:  $(SRC)/ka36.f
	-$(RM) ka36.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka36.f -o ka36.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka36.$(OBJX) check.$(OBJX) $(LIBS) -o ka36.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka36
	ka36.$(EXESUFFIX)

verify: ;

ka36.run: run

