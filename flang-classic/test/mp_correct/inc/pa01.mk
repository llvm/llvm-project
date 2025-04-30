#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
pa01: pa01.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN2) ./a.$(EXESUFFIX) $(LOG)
pa01.$(OBJX): $(SRC)/pa01.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/pa01.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) pa01.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: pa01
run: ;
