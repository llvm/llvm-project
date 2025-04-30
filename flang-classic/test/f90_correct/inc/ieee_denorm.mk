#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee_denorm  ########


ieee_denorm: ieee_denorm.$(OBJX)

ieee_denorm.$(OBJX):  $(SRC)/ieee_denorm.f90
	-$(RM) ieee_denorm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee_denorm.f90 -o ieee_denorm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee_denorm.$(OBJX) check.$(OBJX) $(LIBS) -o ieee_denorm.$(EXESUFFIX)


ieee_denorm.run: ieee_denorm.$(OBJX)
	@echo ------------------------------------ executing test ieee_denorm
	ieee_denorm.$(EXESUFFIX)

verify: ;
build: ieee_denorm.$(OBJX) ;
run: ieee_denorm.run ;
