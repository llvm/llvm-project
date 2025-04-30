#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka45  ########


ka45: run
	

build:  $(SRC)/ka45.f
	-$(RM) ka45.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka45.f -o ka45.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka45.$(OBJX) check.$(OBJX) $(LIBS) -o ka45.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka45
	ka45.$(EXESUFFIX)

verify: ;

ka45.run: run

