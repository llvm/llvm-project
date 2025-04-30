#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test assign_value_array ########


assign_value_array: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/assign_value_array.f08 fcheck.$(OBJX)
	-$(RM) assign_value_array.$(EXESUFFIX) assign_value_array.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/assign_value_array.f08 -o assign_value_array.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) assign_value_array.$(OBJX) fcheck.$(OBJX) $(LIBS) -o assign_value_array.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test assign_value_array
	assign_value_array.$(EXESUFFIX)

verify: ;

assign_value_array.run: run

