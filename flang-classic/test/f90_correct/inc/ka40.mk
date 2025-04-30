#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka40  ########


ka40: run
	

build:  $(SRC)/ka40.f
	-$(RM) ka40.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka40.f -o ka40.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka40.$(OBJX) check.$(OBJX) $(LIBS) -o ka40.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka40
	ka40.$(EXESUFFIX)

verify: ;

ka40.run: run

