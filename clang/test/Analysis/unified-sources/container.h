class ContainerInHeaderFile {
  class Iterator {
  };

public:
  Iterator begin() const;
  Iterator end() const;

  int method() { return 0; }
};
