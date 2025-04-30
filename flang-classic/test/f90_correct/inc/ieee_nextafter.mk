#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test ieee_nextafter  ########


ieee_nextafter: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/ieee_nextafter.f08 fcheck.$(OBJX)
	-$(RM) ieee_nextafter.$(EXESUFFIX) ieee_nextafter.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee_nextafter.f08 -o ieee_nextafter.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee_nextafter.$(OBJX) fcheck.$(OBJX) $(LIBS) -o ieee_nextafter.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ieee_nextafter
	ieee_nextafter.$(EXESUFFIX)

verify: ;

ieee_nextafter.run: run

