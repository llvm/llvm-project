module m
  public
  !ERROR: The default accessibility of this module has already been declared
  private
end

subroutine s
  !ERROR: PUBLIC statement may only appear in the specification part of a module
  public
end
